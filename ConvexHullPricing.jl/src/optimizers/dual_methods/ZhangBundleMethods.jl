function DynamicLevelMethod(
    instance,
    initial_prices,
    stop::StoppingCriterion,
    alpha_init::Float64;
    return_bounds::Bool = false,
    verbose::Int = -1,
    rho_upper::Float64 = 0.5,   # P in the manuscript subroutine
    omega::Float64 = 0.5,       # ω in the manuscript subroutine (must be in (0,1))
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
    set_optimizer_attributes(model_projection, "MIPGap" => 0, "MIPGapAbs" => 0)
    Proj_Price =
        @variable(model_projection, [1:T], lower_bound = BLM_LOWER, upper_bound = BLM_UPPER)
    Proj_t = @variable(model_projection)
    level_constr = nothing

    alpha = alpha_init
    level_prev = nothing   # previous level T_j
    ub_at_prev = nothing   # UB associated with level_prev, i.e., L̂_{j-1} used in I_P and I_A

    time_vector = [0.0]
    idx = 1

    while _blm_continue(stop, idx, time_vector[end], UpperBound - LowerBound)
        verbose > 0 &&
            @info "[BLM: Iteration $idx; UB=$UpperBound, LB=$LowerBound, gap=$(UpperBound - LowerBound), alpha=$alpha]"
        it_time = @elapsed begin
            _ot = @elapsed begin
                fun_oracle, grad_oracle = exact_oracle(oracle, iterates[idx])
            end
            push!(oracle_times, _ot)
            push!(fun_iterates, fun_oracle)
            fun_oracle, grad_oracle = negate_for_minimization(fun_oracle, grad_oracle)

            UpperBound = min(UpperBound, fun_oracle)

            if !isnothing(level_prev) && !isnothing(ub_at_prev)
                I_P = ub_at_prev - level_prev
                I_A = ub_at_prev - UpperBound
                if I_A > 0 && I_P > 0
                    r = I_A / I_P
                    alpha =
                        (r <= rho_upper) ? (1.0 - omega * (1.0 - alpha)) : (omega * alpha)
                end
            end

            @constraint(
                model_update_lb,
                fun_oracle + dot(grad_oracle, VarPrice - iterates[idx]) <= Vart,
            )
            @objective(model_update_lb, Min, Vart)
            optimize!(model_update_lb)
            LowerBound = objective_value(model_update_lb)::Float64

            LevelSet = LowerBound + alpha * (UpperBound - LowerBound)
            level_prev = LevelSet
            ub_at_prev = UpperBound

            @constraint(
                model_projection,
                fun_oracle + dot(grad_oracle, Proj_Price - iterates[idx]) <= Proj_t,
            )

            if !isnothing(level_constr)
                delete(model_projection, level_constr)
            end
            level_constr = @constraint(model_projection, Proj_t <= LevelSet)

            @objective(
                model_projection,
                Min,
                sum((Proj_Price[t] - iterates[idx][t])^2 for t = 1:T),
            )
            optimize!(model_projection)

            if is_solved_and_feasible(model_projection)
                push!(iterates, value.(Proj_Price))
            else
                push!(iterates, iterates[idx])
            end
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
