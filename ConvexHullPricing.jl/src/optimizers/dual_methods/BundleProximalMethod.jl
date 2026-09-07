# ─────────────────────────────────────────────────────────────────────────────
# Bundle Proximal Methods  (see Lemaréchal et al., 1995)
# ─────────────────────────────────────────────────────────────────────────────
# Price box bounds for bundle proximal subproblems
const BPM_LOWER = -500.0
const BPM_UPPER = 3000.0

# ── Bundle Proximal Level Method ────────────────────────────────────────────
# Unifies: BundleProximalLevelMethod, tBundleProximalLevelMethod, tBPLM, tSmoothBPLM
#
# Key options:
#   `project_to_best`      — project towards the best-so-far iterate (true)
#                            or the current iterate (false). Default: true.
#   `smoothing_parameter`  — if set, uses translate_smooth_oracle and
#                            recomputes exact values at the end.
function BundleProximalLevelMethod(
    instance,
    initial_prices,
    stop::StoppingCriterion,
    α;
    project_to_best::Bool = true,
    smoothing_parameter = nothing,
    smooth_oracle_fn = exact_translate_smooth_oracle,
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

    use_smooth = !isnothing(smoothing_parameter)
    shift = use_smooth ? GetShift(instance) : nothing

    # Persistent cutting-plane model
    model_update_lb = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV[]))
    set_silent(model_update_lb)
    set_optimizer_attributes(
        model_update_lb,
        "MIPGap" => 0,
        "MIPGapAbs" => 1e-8,
        "Method" => 1,
    )
    VarPrice =
        @variable(model_update_lb, [1:T], lower_bound = BPM_LOWER, upper_bound = BPM_UPPER)
    Vart = @variable(model_update_lb)

    model_projection = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV[]))
    set_silent(model_projection)
    set_optimizer_attribute(model_projection, "Method", 2)
    VarPi_proj =
        @variable(model_projection, [1:T], lower_bound=BPM_LOWER, upper_bound=BPM_UPPER)
    VarT_proj = @variable(model_projection)
    level_con = nothing

    newGap = Inf
    newLevel = Inf
    BestPoint = initial_prices
    time_vector = [0.0]
    i = 1

    while _blm_continue(stop, i, time_vector[end], UpperBound - LowerBound)
        verbose > 0 &&
            @info "[BPLM: Iteration $i; UB=$UpperBound, LB=$LowerBound, gap=$(UpperBound - LowerBound)]"
        it_time = @elapsed begin
            # Oracle call
            _ot = @elapsed begin
                if use_smooth
                    fun_oracle, grad_oracle = smooth_oracle_fn(
                        instance,
                        iterates[i],
                        smoothing_parameter,
                        shift,
                    )
                else
                    fun_oracle, grad_oracle = exact_oracle(oracle, iterates[i])
                end
            end
            push!(oracle_times, _ot)
            push!(fun_iterates, fun_oracle)
            fun_oracle, grad_oracle = negate_for_minimization(fun_oracle, grad_oracle)

            if fun_oracle < UpperBound
                BestPoint = iterates[i]
            end
            if UpperBound > fun_oracle
                UpperBound = fun_oracle
            end
            push!(UpperBounds, UpperBound)

            # Update cutting-plane lower bound
            @constraint(
                model_update_lb,
                fun_oracle + dot(grad_oracle, VarPrice - iterates[i]) <= Vart,
            )
            @objective(model_update_lb, Min, Vart)
            optimize!(model_update_lb)
            LowerBound = objective_value(model_update_lb)::Float64
            push!(LowerBounds, LowerBound)

            LevelSet = LowerBound + α * (UpperBound - LowerBound)

            # Proximal level update
            if UpperBound - LowerBound >= (1 - α) * newGap
                newLevel = min(LevelSet, newLevel)
            else
                newLevel = LevelSet
                newGap = UpperBound - LowerBound
            end
            newLevel = max(newLevel, LowerBound)   # guard: level set must be non-empty

            # ── Add cut to projection model; update level and center; solve ─
            proj_center = project_to_best ? BestPoint : iterates[i]
            @constraint(
                model_projection,
                fun_oracle + dot(grad_oracle, VarPi_proj - iterates[i]) <= VarT_proj
            )
            isnothing(level_con) || delete(model_projection, level_con)
            level_con = @constraint(model_projection, VarT_proj <= newLevel)
            @objective(
                model_projection,
                Min,
                0.5 * sum((VarPi_proj[j] - proj_center[j])^2 for j = 1:T)
            )
            optimize!(model_projection)
            pi_next =
                is_solved_and_feasible(model_projection) ? value.(VarPi_proj) : iterates[i]
            push!(iterates, pi_next)
        end
        push!(time_vector, it_time + time_vector[end])
        i += 1
        _lstar_reached(stop, fun_iterates[end]) && break
    end

    # Recompute true oracle values when smooth oracle was used
    if use_smooth
        fun_iterates = Float64[exact_oracle(oracle, ρ)[1] for ρ in iterates]
    end

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

# ── Multi-cut Bundle Proximal Level Method ────────────────────────────────
# Multi-cut variant of BundleProximalLevelMethod: under-approximates each
# L_g separately with per-generator epigraph variables.
# See: Stevens & Papavasiliou (2022), Andrianesis et al. (2021).
function MulticutBundleProximalLevelMethod(
    instance,
    initial_prices,
    stop::StoppingCriterion,
    α;
    project_to_best::Bool = true,
    smoothing_parameter = nothing,
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

    use_smooth = !isnothing(smoothing_parameter)
    shift = use_smooth ? GetShift(instance) : nothing

    # Persistent cutting-plane model
    model_update_lb = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV[]))
    set_silent(model_update_lb)
    set_optimizer_attributes(
        model_update_lb,
        "MIPGap" => 0,
        "MIPGapAbs" => 1e-8,
        "Method" => 1,
    )
    VarPrice =
        @variable(model_update_lb, [1:T], lower_bound = BPM_LOWER, upper_bound = BPM_UPPER)
    Vart = @variable(model_update_lb, [1:NbGen])

    model_projection = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV[]))
    set_silent(model_projection)
    set_optimizer_attribute(model_projection, "Method", 2)
    VarPi_proj =
        @variable(model_projection, [1:T], lower_bound=BPM_LOWER, upper_bound=BPM_UPPER)
    VarT_proj = @variable(model_projection, [1:NbGen])
    level_con = nothing

    newGap = Inf
    newLevel = Inf
    BestPoint = initial_prices
    time_vector = [0.0]
    i = 1

    while _blm_continue(stop, i, time_vector[end], UpperBound - LowerBound)
        verbose > 0 &&
            @info "[MC-BPLM: Iteration $i; UB=$UpperBound, LB=$LowerBound, gap=$(UpperBound - LowerBound)]"
        it_time = @elapsed begin
            # Oracle call
            _ot = @elapsed begin
                if use_smooth
                    obj_per_gen, grad_per_gen =
                        exact_translate_smooth_oracle_multicut(
                            instance,
                            iterates[i],
                            smoothing_parameter,
                            shift,
                        )
                else
                    obj_per_gen, grad_per_gen = exact_oracle_multicut(oracle, iterates[i])
                end
            end
            push!(oracle_times, _ot)
            fun_oracle = sum(obj_per_gen)
            push!(fun_iterates, fun_oracle)
            # negate for minimization
            obj_per_gen = -obj_per_gen
            grad_per_gen = [-g for g in grad_per_gen]
            fun_oracle = -fun_oracle

            if fun_oracle < UpperBound
                BestPoint = iterates[i]
            end
            if UpperBound > fun_oracle
                UpperBound = fun_oracle
            end
            push!(UpperBounds, UpperBound)

            # ── Add one cut per generator to both models ───────────────────
            for g = 1:NbGen
                @constraint(
                    model_update_lb,
                    obj_per_gen[g] + dot(grad_per_gen[g], VarPrice - iterates[i]) <=
                    Vart[g],
                )
                @constraint(
                    model_projection,
                    obj_per_gen[g] + dot(grad_per_gen[g], VarPi_proj - iterates[i]) <=
                    VarT_proj[g]
                )
            end

            # ── Lower-bound model ──────────────────────────────────────────
            @objective(model_update_lb, Min, sum(Vart))
            optimize!(model_update_lb)
            LowerBound = objective_value(model_update_lb)::Float64
            push!(LowerBounds, LowerBound)

            LevelSet = LowerBound + α * (UpperBound - LowerBound)

            # Proximal level update
            if UpperBound - LowerBound >= (1 - α) * newGap
                newLevel = min(LevelSet, newLevel)
            else
                newLevel = LevelSet
                newGap = UpperBound - LowerBound
            end
            newLevel = max(newLevel, LowerBound)   # guard: level set must be non-empty

            # ── Projection model: update level and center; solve ───────────
            proj_center = project_to_best ? BestPoint : iterates[i]
            isnothing(level_con) || delete(model_projection, level_con)
            level_con = @constraint(model_projection, sum(VarT_proj) <= newLevel)
            @objective(
                model_projection,
                Min,
                0.5 * sum((VarPi_proj[j] - proj_center[j])^2 for j = 1:T)
            )
            optimize!(model_projection)
            pi_next =
                is_solved_and_feasible(model_projection) ? value.(VarPi_proj) : iterates[i]
            push!(iterates, pi_next)
        end
        push!(time_vector, it_time + time_vector[end])
        i += 1
        _lstar_reached(stop, fun_iterates[end]) && break
    end

    # Recompute true oracle values when smooth oracle was used
    if use_smooth
        fun_iterates = Float64[exact_oracle(oracle, ρ)[1] for ρ in iterates]
    end

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
# ── Preconditioned Bundle Proximal Level Method ───────────────────────────────
# K-cluster-cut BPLM with an EMA-preconditioned projection metric.
# Generators are clustered once (by peak-gradient time bin) into K groups via
# make_peak_time_clusters; one epigraph variable and one cut per cluster replace
# the per-generator structure.
#
# Projection metric: ∑ wₜ (p_t - centre_t)², wₜ = 1/(rmsₜ+ε)².
# This is an *inverse* preconditioner: coordinates with large historical RMS
# gradients get small quadratic penalties and therefore large projection steps
# (KKT displacement ∝ g[j]·(rms[j]+ε)²).  Empirically this aggressive
# scaling in high-gradient directions converges much faster than the RMSProp
# convention (wₜ = rmsₜ+ε), which damps those same directions.
# K defaults to 50; mem defaults to 0 (no cut dropping).
function PreconditionedProximalLevelMethod(
    instance,
    initial_prices,
    stop::StoppingCriterion,
    α;
    project_to_best::Bool = true,
    return_bounds::Bool = false,
    verbose::Int = -1,
    mem::Int = 0,
    K::Int = 50,
    initial_ub_lstar::Float64 = Inf,
    preconditioner::PreconditionerConfig = ClippedInversePrecond(),
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

    precond_state = init_precond_state(preconditioner, T)
    w = ones(Float64, T)

    model_update_lb = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV[]))
    set_silent(model_update_lb)
    set_optimizer_attributes(
        model_update_lb,
        "MIPGap" => 0,
        "MIPGapAbs" => 1e-8,
        "Method" => 1,
    )
    VarPrice =
        @variable(model_update_lb, [1:T], lower_bound = BPM_LOWER, upper_bound = BPM_UPPER)
    Vart = @variable(model_update_lb, [1:K])

    model_projection = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV[]))
    set_silent(model_projection)
    set_optimizer_attribute(model_projection, "Method", 2)
    VarPi_proj =
        @variable(model_projection, [1:T], lower_bound=BPM_LOWER, upper_bound=BPM_UPPER)
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

    newGap = Inf
    newLevel = Inf
    BestPoint = initial_prices
    time_vector = [0.0]
    i = 1

    while _blm_continue(stop, i, time_vector[end], UpperBound - LowerBound)
        verbose > 0 &&
            @info "[PC-BPLM: Iteration $i; UB=$UpperBound, LB=$LowerBound, gap=$(UpperBound - LowerBound)]"
        it_time = @elapsed begin
            _ot = @elapsed begin
                obj_per_gen, grad_per_gen = exact_oracle_multicut(oracle, iterates[i])
            end
            push!(oracle_times, _ot)
            fun_oracle = sum(obj_per_gen)
            push!(fun_iterates, fun_oracle)
            obj_per_gen = -obj_per_gen
            grad_per_gen = [-g for g in grad_per_gen]
            fun_oracle = -fun_oracle

            if fun_oracle < UpperBound
                BestPoint = iterates[i]
            end
            if UpperBound > fun_oracle
                UpperBound = fun_oracle
            end
            push!(UpperBounds, UpperBound)

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

            # ── Compute aggregate gradient and preconditioner weights ─────
            agg_grad = zeros(Float64, T)
            @inbounds for k = 1:K
                agg_grad .+= gradC[k]
            end
            compute_weights!(w, preconditioner, precond_state, agg_grad, gradC, K, T)

            # ── One cut per cluster ────────────────────────────────────────
            if mem > 0
                lb_refs = [
                    @constraint(
                        model_update_lb,
                        objC[k] + dot(gradC[k], VarPrice - iterates[i]) <= Vart[k]
                    ) for k = 1:K
                ]
                proj_refs = [
                    @constraint(
                        model_projection,
                        objC[k] + dot(gradC[k], VarPi_proj - iterates[i]) <= VarT_proj[k]
                    ) for k = 1:K
                ]
                push!(cut_buffer, (objC, gradC, iterates[i], lb_refs))
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
                    # Replace aggregate cut constraints in both models
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
                        objC[k] + dot(gradC[k], VarPrice - iterates[i]) <= Vart[k],
                    )
                    @constraint(
                        model_projection,
                        objC[k] + dot(gradC[k], VarPi_proj - iterates[i]) <= VarT_proj[k]
                    )
                end
            end

            # ── Lower-bound model ──────────────────────────────────────────
            @objective(model_update_lb, Min, sum(Vart))
            optimize!(model_update_lb)
            LowerBound = max(LowerBound, objective_value(model_update_lb))
            push!(LowerBounds, LowerBound)

            LevelSet = LowerBound + α * (UpperBound - LowerBound)

            # ── Proximal level update ──────────────────────────────────────
            if UpperBound - LowerBound >= (1 - α) * newGap
                newLevel = min(LevelSet, newLevel)
            else
                newLevel = LevelSet
                newGap = UpperBound - LowerBound
            end
            newLevel = max(newLevel, LowerBound)

            # ── Projection model: update level and center; solve ───────────
            proj_center = project_to_best ? BestPoint : iterates[i]
            isnothing(level_con) || delete(model_projection, level_con)
            level_con = @constraint(model_projection, sum(VarT_proj) <= newLevel)
            @objective(
                model_projection,
                Min,
                0.5 * sum(w[j] * (VarPi_proj[j] - proj_center[j])^2 for j = 1:T)
            )
            optimize!(model_projection)
            pi_next =
                is_solved_and_feasible(model_projection) ? value.(VarPi_proj) : iterates[i]
            push!(iterates, pi_next)
        end
        push!(time_vector, it_time + time_vector[end])
        i += 1
        _lstar_reached(stop, fun_iterates[end]) && break
    end

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
