# Price box bounds for the level-set subproblems (wider than the main PCD/PCU)
const BLM_LOWER = -500.0
const BLM_UPPER = 3000.0

function BundleLevelMethod(
    instance,
    initial_prices,
    stop::StoppingCriterion,
    alpha;
    return_bounds::Bool = false,
    verbose::Int = -1,
    initial_ub_lstar::Float64 = Inf,
)
    oracle = build_oracle_subproblems(instance)
    T = length(instance.Load)
    iterates = [initial_prices]
    fun_iterates = Float64[]
    oracle_times = Float64[]

    UpperBound = Inf
    LowerBound = isinf(initial_ub_lstar) ? -Inf : -initial_ub_lstar
    UpperBounds = Float64[]
    LowerBounds = Float64[]

    model_update_lb = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV[]))
    set_silent(model_update_lb)
    set_optimizer_attributes(
        model_update_lb,
        "MIPGap" => 0,
        "MIPGapAbs" => 0,
        "Method" => 1,
    )
    VarPrice =
        @variable(model_update_lb, [1:T], lower_bound = BLM_LOWER, upper_bound = BLM_UPPER)
    Vart = @variable(model_update_lb)

    model_projection = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV[]))
    set_silent(model_projection)
    set_optimizer_attribute(model_projection, "Method", 2)
    VarPi_proj =
        @variable(model_projection, [1:T], lower_bound=BLM_LOWER, upper_bound=BLM_UPPER)
    VarT_proj = @variable(model_projection)
    level_con = nothing

    time_vector = [0.0]
    idx = 1

    while _blm_continue(stop, idx, time_vector[end], UpperBound - LowerBound)
        verbose > 0 &&
            @info "[BLM: Iteration $idx; UB=$UpperBound, LB=$LowerBound, gap=$(UpperBound - LowerBound)]"
        it_time = @elapsed begin
            _ot = @elapsed begin
                fun_oracle, grad_oracle = exact_oracle(oracle, iterates[idx])
            end
            push!(oracle_times, _ot)
            push!(fun_iterates, fun_oracle)
            fun_oracle, grad_oracle = negate_for_minimization(fun_oracle, grad_oracle)

            if UpperBound > fun_oracle
                UpperBound = fun_oracle
            end

            # ── Update lower-bound model ───────────────────────────────────
            @constraint(
                model_update_lb,
                fun_oracle + dot(grad_oracle, VarPrice - iterates[idx]) <= Vart,
            )
            @objective(model_update_lb, Min, Vart)
            optimize!(model_update_lb)
            LowerBound = objective_value(model_update_lb)::Float64

            LevelSet = LowerBound + alpha * (UpperBound - LowerBound)

            # ── Add cut to projection model; update level; solve ──────────
            @constraint(
                model_projection,
                fun_oracle + dot(grad_oracle, VarPi_proj - iterates[idx]) <= VarT_proj
            )
            isnothing(level_con) || delete(model_projection, level_con)
            level_con = @constraint(model_projection, VarT_proj <= LevelSet)
            c = iterates[idx]
            @objective(
                model_projection,
                Min,
                0.5 * sum((VarPi_proj[j] - c[j])^2 for j = 1:T)
            )
            optimize!(model_projection)
            pi_next =
                is_solved_and_feasible(model_projection) ? value.(VarPi_proj) :
                iterates[idx]
            push!(iterates, pi_next)
        end
        push!(time_vector, it_time + time_vector[end])
        push!(UpperBounds, UpperBound)
        push!(LowerBounds, LowerBound)
        idx += 1
        _lstar_reached(stop, fun_iterates[end]) && break
    end

    verbose > 0 &&
        @info "UB = $UpperBound, LB = $LowerBound, UB-LB = $(UpperBound - LowerBound)"
    if return_bounds
        return last(iterates),
        iterates,
        fun_iterates,
        time_vector,
        oracle_times,
        UpperBounds,
        LowerBounds
    end
    return last(iterates), iterates, fun_iterates, time_vector, oracle_times
end

_blm_continue(c::GapTolerance, idx, elapsed, gap) = gap >= c.tolerance
_blm_continue(c::GapToleranceWithBudget, idx, elapsed, gap) =
    elapsed <= c.max_seconds && gap >= c.tolerance
_blm_continue(c::StoppingCriterion, idx, elapsed, _) = should_continue(c, idx, elapsed)

# ── Preconditioned Level Method ──────────────────────────────────────────────
# K-cluster-cut BLM with an EMA-preconditioned projection metric.
# Generators are clustered once (by peak-gradient time bin) into K groups via
# make_peak_time_clusters; one epigraph variable and one cut per cluster replace
# the per-generator structure.
#
# Projection metric: ∑ wₜ (p_t - pₜ_curr)², wₜ = 1/(rmsₜ+ε)².
# This is an *inverse* preconditioner: coordinates with large historical RMS
# gradients get small quadratic penalties and therefore large projection steps
# (KKT displacement ∝ g[j]·(rms[j]+ε)²).  Empirically this aggressive
# scaling in high-gradient directions converges much faster than the RMSProp
# convention (wₜ = rmsₜ+ε), which damps those same directions.
# K defaults to 8; mem defaults to 0 (no cut dropping).
function PreconditionedLevelMethod(
    instance,
    initial_prices,
    stop::StoppingCriterion,
    alpha;
    return_bounds::Bool = false,
    verbose::Int = -1,
    mem::Int = 0,
    K::Int = 8,
    initial_ub_lstar::Float64 = Inf,
)
    oracle = build_oracle_subproblems(instance)
    precond_eps::Float64 = 1e-8  # numerical floor in weights
    precond_beta::Float64 = 0.9   # EMA smoothing (higher → smoother)
    T = length(instance.Load)
    iterates = [initial_prices]
    fun_iterates = Float64[]
    oracle_times = Float64[]

    UpperBound = Inf
    LowerBound = isinf(initial_ub_lstar) ? -Inf : -initial_ub_lstar
    UpperBounds = Float64[]
    LowerBounds = Float64[]

    rms_grad = ones(Float64, T)  # running RMS of aggregate subgradient

    model_update_lb = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV[]))
    set_silent(model_update_lb)
    set_optimizer_attributes(
        model_update_lb,
        "MIPGap" => 0,
        "MIPGapAbs" => 0,
        "Method" => 1,
    )
    VarPrice =
        @variable(model_update_lb, [1:T], lower_bound = BLM_LOWER, upper_bound = BLM_UPPER)
    Vart = @variable(model_update_lb, [1:K])

    model_projection = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV[]))
    set_silent(model_projection)
    set_optimizer_attribute(model_projection, "Method", 2)
    VarPi_proj =
        @variable(model_projection, [1:T], lower_bound=BLM_LOWER, upper_bound=BLM_UPPER)
    VarT_proj = @variable(model_projection, [1:K])
    level_con = nothing

    # ── Memory-limited bundle state (used only when mem > 0) ─────────────────
    # Each entry in cut_buffer: (objC, gradC, iterate, lb_refs)
    # agg_const[k]    = Σ (objC[k] - dot(gradC[k], π)) for all dropped iterations
    # agg_grad_sum[k] = Σ gradC[k]  for all dropped iterations
    cut_buffer = Vector{Any}()
    agg_count = 0
    agg_const = zeros(Float64, K)
    agg_grad_sum = [zeros(Float64, T) for _ = 1:K]
    agg_cuts_lb = nothing
    proj_cut_refs_buffer = Vector{Any}()   # projection cut refs per buffer entry
    agg_proj_cuts = nothing         # refs to aggregate projection cuts

    # Clusters built once after the first oracle call
    clusters = nothing

    time_vector = [0.0]
    idx = 1

    while _blm_continue(stop, idx, time_vector[end], UpperBound - LowerBound)
        verbose > 0 &&
            @info "[PC-BLM: Iteration $idx; UB=$UpperBound, LB=$LowerBound, gap=$(UpperBound - LowerBound)]"
        it_time = @elapsed begin
            _ot = @elapsed begin
                obj_per_gen, grad_per_gen = exact_oracle_multicut(oracle, iterates[idx])
            end
            push!(oracle_times, _ot)
            fun_oracle = sum(obj_per_gen)
            push!(fun_iterates, fun_oracle)
            obj_per_gen = -obj_per_gen
            grad_per_gen = [-g for g in grad_per_gen]
            fun_oracle = -fun_oracle

            if UpperBound > fun_oracle
                UpperBound = fun_oracle
            end

            # ── Build clusters once from the first oracle output ───────────
            if isnothing(clusters)
                clusters = make_peak_time_clusters(grad_per_gen, T, K)
            end

            # ── Aggregate oracle results into K cluster cuts ───────────────
            objC = zeros(Float64, K)
            gradC = [zeros(Float64, T) for _ = 1:K]
            for k = 1:K
                for g in clusters[k]
                    objC[k] += obj_per_gen[g]
                    gradC[k] .+= grad_per_gen[g]
                end
            end

            # ── EMA preconditioning ────────────────────────────────────────
            agg_grad = zeros(Float64, T)
            @inbounds for k = 1:K
                agg_grad .+= gradC[k]
            end
            @inbounds for t = 1:T
                s2 = precond_beta * (rms_grad[t]^2) + (1.0 - precond_beta) * (agg_grad[t]^2)
                rms_grad[t] = sqrt(s2)
            end
            w = similar(rms_grad)
            @inbounds for t = 1:T
                w[t] = 1.0 / (rms_grad[t] + precond_eps)^2
            end

            # ── One cut per cluster in both models ─────────────────────────
            if mem > 0
                lb_refs = [
                    @constraint(
                        model_update_lb,
                        objC[k] + dot(gradC[k], VarPrice - iterates[idx]) <= Vart[k]
                    ) for k = 1:K
                ]
                proj_refs = [
                    @constraint(
                        model_projection,
                        objC[k] + dot(gradC[k], VarPi_proj - iterates[idx]) <= VarT_proj[k]
                    ) for k = 1:K
                ]
                push!(cut_buffer, (objC, gradC, iterates[idx], lb_refs))
                push!(proj_cut_refs_buffer, proj_refs)

                if length(cut_buffer) > mem
                    old_objC, old_gradC, old_π, old_lb_refs = popfirst!(cut_buffer)
                    old_proj_refs = popfirst!(proj_cut_refs_buffer)
                    # Remove the oldest iteration's constraints from both models
                    for k = 1:K
                        delete(model_update_lb, old_lb_refs[k])
                        delete(model_projection, old_proj_refs[k])
                    end
                    # Accumulate dropped cut data into aggregate
                    agg_count += 1
                    for k = 1:K
                        agg_const[k] += old_objC[k] - dot(old_gradC[k], old_π)
                        agg_grad_sum[k] .+= old_gradC[k]
                    end
                    # Replace aggregate lb cut constraints
                    if !isnothing(agg_cuts_lb)
                        for c in agg_cuts_lb
                            ;
                            delete(model_update_lb, c);
                        end
                    end
                    if !isnothing(agg_proj_cuts)
                        for c in agg_proj_cuts
                            ;
                            delete(model_projection, c);
                        end
                    end
                    inv_n = 1.0 / agg_count
                    agg_cuts_lb = [
                        @constraint(
                            model_update_lb,
                            agg_const[k] * inv_n + dot(agg_grad_sum[k] * inv_n, VarPrice) <= Vart[k]
                        ) for k = 1:K
                    ]
                    agg_proj_cuts = [
                        @constraint(
                            model_projection,
                            agg_const[k] * inv_n +
                            dot(agg_grad_sum[k] * inv_n, VarPi_proj) <=
                            VarT_proj[k]
                        ) for k = 1:K
                    ]
                end
            else
                for k = 1:K
                    @constraint(
                        model_update_lb,
                        objC[k] + dot(gradC[k], VarPrice - iterates[idx]) <= Vart[k],
                    )
                    @constraint(
                        model_projection,
                        objC[k] + dot(gradC[k], VarPi_proj - iterates[idx]) <= VarT_proj[k]
                    )
                end
            end

            # ── Lower-bound model ──────────────────────────────────────────
            @objective(model_update_lb, Min, sum(Vart))
            optimize!(model_update_lb)
            LowerBound = max(LowerBound, objective_value(model_update_lb))

            LevelSet = LowerBound + alpha * (UpperBound - LowerBound)

            # ── Projection model: update level and objective; solve ─────────
            isnothing(level_con) || delete(model_projection, level_con)
            level_con = @constraint(model_projection, sum(VarT_proj) <= LevelSet)
            c = iterates[idx]
            @objective(
                model_projection,
                Min,
                0.5 * sum(w[j] * (VarPi_proj[j] - c[j])^2 for j = 1:T)
            )
            optimize!(model_projection)
            pi_next =
                is_solved_and_feasible(model_projection) ? value.(VarPi_proj) :
                iterates[idx]
            push!(iterates, pi_next)
        end
        push!(time_vector, it_time + time_vector[end])
        push!(UpperBounds, UpperBound)
        push!(LowerBounds, LowerBound)
        idx += 1
        _lstar_reached(stop, fun_iterates[end]) && break
    end

    verbose > 0 &&
        @info "UB = $UpperBound, LB = $LowerBound, UB-LB = $(UpperBound - LowerBound)"
    if return_bounds
        return last(iterates),
        iterates,
        fun_iterates,
        time_vector,
        oracle_times,
        UpperBounds,
        LowerBounds
    end
    return last(iterates), iterates, fun_iterates, time_vector, oracle_times
end

function MulticutBundleLevelMethod(
    instance,
    initial_prices,
    stop::StoppingCriterion,
    alpha;
    return_bounds::Bool = false,
    verbose::Int = -1,
    initial_ub_lstar::Float64 = Inf,
)
    oracle = build_oracle_subproblems(instance)
    T = length(instance.Load)
    NbGen = unpack_instance(instance).NbGen
    iterates = [initial_prices]
    fun_iterates = Float64[]
    oracle_times = Float64[]

    UpperBound = Inf
    LowerBound = isinf(initial_ub_lstar) ? -Inf : -initial_ub_lstar
    UpperBounds = Float64[]
    LowerBounds = Float64[]

    model_update_lb = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV[]))
    set_silent(model_update_lb)
    set_optimizer_attributes(
        model_update_lb,
        "MIPGap" => 0,
        "MIPGapAbs" => 0,
        "Method" => 1,
    )
    VarPrice =
        @variable(model_update_lb, [1:T], lower_bound = BLM_LOWER, upper_bound = BLM_UPPER)
    Vart = @variable(model_update_lb, [1:NbGen])

    model_projection = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV[]))
    set_silent(model_projection)
    set_optimizer_attribute(model_projection, "Method", 2)
    VarPi_proj =
        @variable(model_projection, [1:T], lower_bound=BLM_LOWER, upper_bound=BLM_UPPER)
    VarT_proj = @variable(model_projection, [1:NbGen])
    level_con = nothing

    time_vector = [0.0]
    idx = 1

    while _blm_continue(stop, idx, time_vector[end], UpperBound - LowerBound)
        verbose > 0 &&
            @info "[MC-BLM: Iteration $idx; UB=$UpperBound, LB=$LowerBound, gap=$(UpperBound - LowerBound)]"
        it_time = @elapsed begin
            _ot = @elapsed begin
                obj_per_gen, grad_per_gen = exact_oracle_multicut(oracle, iterates[idx])
            end
            push!(oracle_times, _ot)
            fun_oracle = sum(obj_per_gen)
            push!(fun_iterates, fun_oracle)
            # negate for minimization
            obj_per_gen = -obj_per_gen
            grad_per_gen = [-g for g in grad_per_gen]
            fun_oracle = -fun_oracle

            if UpperBound > fun_oracle
                UpperBound = fun_oracle
            end

            # ── Update both models: one cut per generator ──────────────────
            for g = 1:NbGen
                @constraint(
                    model_update_lb,
                    obj_per_gen[g] + dot(grad_per_gen[g], VarPrice - iterates[idx]) <=
                    Vart[g],
                )
                @constraint(
                    model_projection,
                    obj_per_gen[g] + dot(grad_per_gen[g], VarPi_proj - iterates[idx]) <=
                    VarT_proj[g]
                )
            end
            @objective(model_update_lb, Min, sum(Vart))
            optimize!(model_update_lb)
            LowerBound = objective_value(model_update_lb)::Float64

            LevelSet = LowerBound + alpha * (UpperBound - LowerBound)

            # ── Projection model: update level and objective; solve ─────────
            isnothing(level_con) || delete(model_projection, level_con)
            level_con = @constraint(model_projection, sum(VarT_proj) <= LevelSet)
            c = iterates[idx]
            @objective(
                model_projection,
                Min,
                0.5 * sum((VarPi_proj[j] - c[j])^2 for j = 1:T)
            )
            optimize!(model_projection)
            pi_next =
                is_solved_and_feasible(model_projection) ? value.(VarPi_proj) :
                iterates[idx]
            push!(iterates, pi_next)
        end
        push!(time_vector, it_time + time_vector[end])
        push!(UpperBounds, UpperBound)
        push!(LowerBounds, LowerBound)
        idx += 1
        _lstar_reached(stop, fun_iterates[end]) && break
    end

    verbose > 0 &&
        @info "UB = $UpperBound, LB = $LowerBound, UB-LB = $(UpperBound - LowerBound)"
    if return_bounds
        return last(iterates),
        iterates,
        fun_iterates,
        time_vector,
        oracle_times,
        UpperBounds,
        LowerBounds
    end
    return last(iterates), iterates, fun_iterates, time_vector, oracle_times
end
