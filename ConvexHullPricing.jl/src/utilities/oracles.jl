
function LP_Relaxation(instance)
    p = unpack_instance(instance)

    model, vars = build_joint_uc_model(p; binary = false, balance = true)
    @objective(model, Min, joint_production_cost(p, vars))
    optimize!(model)

    return dual.(vars.loads)
end

"""
    fast_uc_upper_bound(instance; time_limit=10.0) -> Float64

Solve the UC MILP with a short time limit to find a feasible integer schedule.
Returns the objective value (an upper bound on L*) if a feasible solution is found,
or Inf if no feasible solution is found within the time limit.

Any feasible UC schedule has cost C ≥ L* by weak duality, so −C is a valid
lower bound in the negated minimization sense used by bundle methods.
"""
function fast_uc_upper_bound(instance; time_limit::Float64 = 10.0)
    p = unpack_instance(instance)
    model, vars = build_joint_uc_model(p; binary = true, balance = true)
    set_silent(model)
    set_optimizer_attribute(model, "TimeLimit", time_limit)
    set_optimizer_attribute(model, "MIPGap", 0.05)
    @objective(model, Min, joint_production_cost(p, vars))
    optimize!(model)
    has_values(model) || return Inf
    return objective_value(model)
end

function GetShift(instance)
    p = unpack_instance(instance)
    T, NbGen = p.T, p.NbGen

    model, vars = build_joint_uc_model(p; binary = false, balance = true)
    @objective(model, Min, joint_production_cost(p, vars))
    optimize!(model)

    shiftU = [value(vars.Varu[g, t]) for g = 1:NbGen, t = 1:T]
    shiftV = [value(vars.Varv[g, t]) for g = 1:NbGen, t = 1:T]
    shiftW = [value(vars.Varw[g, t]) for g = 1:NbGen, t = 1:T]
    shiftP = [value(vars.Varp[g, t]) for g = 1:NbGen, t = 1:T]
    shiftL = [value(vars.VarL[t]) for t = 1:T]

    return (shiftU, shiftV, shiftW, shiftP, shiftL)
end


"""
    ExactOracleCache

Pre-built per-generator subproblem models for `exact_oracle` and
`exact_oracle_multicut`.  Build once with `build_oracle_subproblems(instance)`
and pass in place of `instance` to avoid reconstructing Gurobi models on every
oracle call; Gurobi warm-starts each solve from the previous solution.
"""
struct ExactOracleCache
    models::Vector{JuMP.Model}
    vars::Vector{Any}
    p::Any
end

"""
    build_oracle_subproblems(instance) -> ExactOracleCache

Pre-build one Gurobi subproblem model per generator for the exact Lagrangian
oracle.  Pass the returned cache to `exact_oracle` or `exact_oracle_multicut`
instead of `instance` to enable warm-starting across iterations.
"""
function build_oracle_subproblems(instance)
    p = unpack_instance(instance)
    models = Vector{JuMP.Model}(undef, p.NbGen)
    vars_vec = Vector{Any}(undef, p.NbGen)
    for gen = 1:p.NbGen
        model, vars = build_single_gen_subproblem(gen, p; mip_gap = 1e-8, mip_gap_abs = 0)
        models[gen] = model
        vars_vec[gen] = vars
    end
    return ExactOracleCache(models, vars_vec, p)
end

function exact_oracle(cache::ExactOracleCache, prices)
    p = cache.p
    nthreads_max = Threads.maxthreadid()
    obj_by_thread = zeros(Float64, nthreads_max)
    grad_by_thread = [zeros(Float64, p.T) for _ = 1:nthreads_max]

    Threads.@threads for gen = 1:p.NbGen
        tid = Threads.threadid()
        model = cache.models[gen]
        vars = cache.vars[gen]
        @objective(model, Min, lagrangian_gen_cost(p, gen, vars, prices))
        optimize!(model)

        obj_by_thread[tid] += objective_value(model)
        @inbounds for t = 1:p.T
            grad_by_thread[tid][t] += value(vars.VarL[t]) / p.NbGen - value(vars.Varp[t])
        end
    end

    ObjOracle = sum(obj_by_thread)
    GradOracle = zeros(Float64, p.T)
    for tid = 1:nthreads_max
        @inbounds for t = 1:p.T
            GradOracle[t] += grad_by_thread[tid][t]
        end
    end
    return ObjOracle, GradOracle
end

function exact_oracle_multicut(cache::ExactOracleCache, prices)
    p = cache.p
    obj_per_gen = Vector{Float64}(undef, p.NbGen)
    grad_per_gen = [Vector{Float64}(undef, p.T) for _ = 1:p.NbGen]

    Threads.@threads for gen = 1:p.NbGen
        model = cache.models[gen]
        vars = cache.vars[gen]
        @objective(model, Min, lagrangian_gen_cost(p, gen, vars, prices))
        optimize!(model)

        obj_per_gen[gen] = objective_value(model)
        extract_gen_gradient!(grad_per_gen[gen], vars, p.T, p.NbGen)
    end
    return obj_per_gen, grad_per_gen
end

function exact_oracle(instance, prices)
    p = unpack_instance(instance)

    nthreads_max = Threads.maxthreadid()
    obj_by_thread = zeros(Float64, nthreads_max)
    grad_by_thread = [zeros(Float64, p.T) for _ = 1:nthreads_max]

    Threads.@threads for gen = 1:p.NbGen
        tid = Threads.threadid()
        model, vars = build_single_gen_subproblem(gen, p; mip_gap = 1e-8, mip_gap_abs = 0)
        @objective(model, Min, lagrangian_gen_cost(p, gen, vars, prices))
        optimize!(model)

        obj_by_thread[tid] += objective_value(model)
        @inbounds for t = 1:p.T
            grad_by_thread[tid][t] += value(vars.VarL[t]) / p.NbGen - value(vars.Varp[t])
        end
    end

    ObjOracle = sum(obj_by_thread)
    GradOracle = zeros(Float64, p.T)
    for tid = 1:nthreads_max
        @inbounds for t = 1:p.T
            GradOracle[t] += grad_by_thread[tid][t]
        end
    end
    return ObjOracle, GradOracle
end


function exact_smooth_oracle(instance, prices, smoothing_parameter)
    p = unpack_instance(instance)

    ObjOracle = 0.0
    GradOracle = zeros(p.T)

    for gen = 1:p.NbGen
        model, vars = build_single_gen_subproblem(gen, p)
        prox = origin_prox_term(vars, p.T)
        @objective(
            model,
            Min,
            lagrangian_gen_cost(p, gen, vars, prices) + (smoothing_parameter / 2) * prox
        )
        optimize!(model)

        ObjOracle += objective_value(model)::Float64
        GradOracle += extract_gen_gradient(vars, p.T, p.NbGen)
    end
    return ObjOracle, GradOracle
end

"""
    exact_oracle_multicut(instance, prices)
        -> (obj_per_gen::Vector{Float64}, grad_per_gen::Vector{Vector{Float64}})

Same computation as `exact_oracle`, but returns per-generator values instead
of aggregating.  `obj_per_gen[g]` = L_g(π), `grad_per_gen[g]` = ∇L_g(π).
"""
function exact_oracle_multicut(instance, prices)
    p = unpack_instance(instance)
    obj_per_gen = Vector{Float64}(undef, p.NbGen)
    grad_per_gen = [Vector{Float64}(undef, p.T) for _ = 1:p.NbGen]

    Threads.@threads for gen = 1:p.NbGen
        model, vars = build_single_gen_subproblem(gen, p; mip_gap = 1e-8, mip_gap_abs = 0)
        @objective(model, Min, lagrangian_gen_cost(p, gen, vars, prices))
        optimize!(model)

        obj_per_gen[gen] = objective_value(model)
        extract_gen_gradient!(grad_per_gen[gen], vars, p.T, p.NbGen)
    end
    return obj_per_gen, grad_per_gen
end

"""
    exact_translate_smooth_oracle(instance, prices, ς, shift) -> (obj, grad)

Smoothed translated Lagrangian oracle using:
- Closed-form demand block (eq. 15)
- Per-generator subproblem with linearized binary proximal (eq. 16)
"""
function exact_translate_smooth_oracle(instance, prices, smoothing_parameter, shift)
    p = unpack_instance(instance)
    _, _, _, _, shiftL = shift
    NbGen = p.NbGen

    # Demand block: closed-form (gradient is l_star, not scaled by NbGen)
    obj_demand, grad_demand, _ =
        demand_block_closedform(p, prices, smoothing_parameter, shiftL)

    ObjOracle = obj_demand
    GradOracle = copy(grad_demand)

    for gen = 1:NbGen
        model, vars = build_gen_subproblem(gen, p)
        @objective(
            model,
            Min,
            gen_smoothed_objective(p, gen, vars, prices, smoothing_parameter, shift)
        )
        optimize!(model)

        ObjOracle += objective_value(model)::Float64
        GradOracle += extract_gen_gradient_no_load(vars, p.T)
    end
    return ObjOracle, GradOracle
end

"""
    exact_translate_smooth_oracle_multicut(instance, prices, ς, shift)
        -> (obj_per_gen::Vector{Float64}, grad_per_gen::Vector{Vector{Float64}})

Multi-cut variant of `exact_translate_smooth_oracle`.
`obj_per_gen[1]` and `grad_per_gen[1]` correspond to the demand block.
`obj_per_gen[g+1]` and `grad_per_gen[g+1]` correspond to generator `g`.
"""
function exact_translate_smooth_oracle_multicut(
    instance,
    prices,
    smoothing_parameter,
    shift,
)
    p = unpack_instance(instance)
    _, _, _, _, shiftL = shift
    NbGen = p.NbGen

    # NbGen+1 components: demand block + NbGen generator blocks
    obj_per_gen = zeros(NbGen + 1)
    grad_per_gen = [zeros(p.T) for _ = 1:(NbGen+1)]

    # Demand block
    obj_demand, grad_demand, _ =
        demand_block_closedform(p, prices, smoothing_parameter, shiftL)
    obj_per_gen[1] = obj_demand
    grad_per_gen[1] = copy(grad_demand)

    for gen = 1:NbGen
        model, vars = build_gen_subproblem(gen, p)
        @objective(
            model,
            Min,
            gen_smoothed_objective(p, gen, vars, prices, smoothing_parameter, shift)
        )
        optimize!(model)

        obj_per_gen[gen+1] = objective_value(model)
        grad_per_gen[gen+1] = extract_gen_gradient_no_load(vars, p.T)
    end
    return obj_per_gen, grad_per_gen
end

