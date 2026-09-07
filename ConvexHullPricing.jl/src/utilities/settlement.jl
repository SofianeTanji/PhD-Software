"""
    compute_ch_prices(instance; method=:column_generation, budget=60.0, kwargs...)

Compute convex-hull prices with one call. `method` may be a supported symbol or
a callable with signature `(instance, initial_prices, stop; kwargs...)`.
`budget` may be a `StoppingCriterion`, an integer iteration limit, or a time
budget in seconds.
"""
function compute_ch_prices(
    instance;
    method = :column_generation,
    budget = 60.0,
    initial_prices = nothing,
    eps::Float64 = 1e-6,
    alpha::Float64 = 0.5,
    smoothing_parameter = nothing,
    stepsize::Float64 = 1.0,
    pool_size::Int = 1,
    initial_distance_estimate::Float64 = PC,
    obj_sol = nothing,
    verbose::Int = -1,
    return_result::Bool = false,
    kwargs...,
)
    p = unpack_instance(instance)
    stop = _budget_to_stop(budget)
    prices0 = _price_vector(p, isnothing(initial_prices) ? LP_Relaxation(instance) : initial_prices)

    result = _run_ch_method(
        method,
        instance,
        prices0,
        stop;
        eps,
        alpha,
        smoothing_parameter,
        stepsize,
        pool_size,
        initial_distance_estimate,
        obj_sol,
        verbose,
        kwargs...,
    )
    return return_result ? result : first(result)
end

"""
    solve_market_schedule(instance; mip_gap=0.0, time_limit=nothing)

Solve the centralized unit-commitment problem and return the dispatch used for
settlement.
"""
function solve_market_schedule(instance; mip_gap = 0.0, time_limit = nothing)
    p = unpack_instance(instance)
    model, vars = build_joint_uc_model(p; binary = true, balance = true, mip_gap)
    if !isnothing(time_limit)
        set_optimizer_attribute(model, "TimeLimit", Float64(time_limit))
    end
    @objective(model, Min, joint_production_cost(p, vars))
    optimize!(model)
    _require_values(model, "market schedule")

    production = _extract_gen_time_values(vars.Varp, p.NbGen, p.T)
    commitment = _extract_gen_time_values(vars.Varu, p.NbGen, p.T)
    startup = _extract_gen_time_values(vars.Varv, p.NbGen, p.T)
    shutdown = _extract_gen_time_values(vars.Varw, p.NbGen, p.T)
    served_load = [value(vars.VarL[t]) for t = 1:p.T]
    lost_load = p.L .- served_load
    costs = _unit_production_costs(p, production, commitment, startup)

    return (
        production = production,
        commitment = commitment,
        startup = startup,
        shutdown = shutdown,
        served_load = served_load,
        lost_load = lost_load,
        costs = costs,
        lost_load_cost = p.LostLoad * sum(lost_load),
        total_cost = sum(costs) + p.LostLoad * sum(lost_load),
        objective_value = objective_value(model),
    )
end

"""
    self_schedule(instance, prices)

Compute each generator's profit-maximizing schedule at fixed prices.
"""
function self_schedule(instance, prices)
    p = unpack_instance(instance)
    price_vec = _price_vector(p, prices)

    production = zeros(Float64, p.NbGen, p.T)
    commitment = zeros(Float64, p.NbGen, p.T)
    startup = zeros(Float64, p.NbGen, p.T)
    shutdown = zeros(Float64, p.NbGen, p.T)
    costs = zeros(Float64, p.NbGen)

    for gen = 1:p.NbGen
        sub = build_cg_subproblem(gen, p)
        @objective(
            sub.model,
            Min,
            sub.VarCost - sum(price_vec[t] * sub.Varp[t] for t = 1:p.T)
        )
        optimize!(sub.model)
        _require_values(sub.model, "self-schedule for generator $gen")

        for t = 1:p.T
            production[gen, t] = value(sub.Varp[t])
            commitment[gen, t] = value(sub.Varu[t])
            startup[gen, t] = value(sub.Varv[t])
            shutdown[gen, t] = value(sub.Varw[t])
        end
        costs[gen] = value(sub.VarCost)
    end

    revenues = _unit_revenues(price_vec, production)
    profits = revenues .- costs

    return (
        production = production,
        commitment = commitment,
        startup = startup,
        shutdown = shutdown,
        costs = costs,
        revenues = revenues,
        profits = profits,
    )
end

"""
    unit_uplifts(instance, prices, market_schedule)

Return lost opportunity costs per generator.
"""
function unit_uplifts(instance, prices, market_schedule)
    return _settlement_components(instance, prices, market_schedule).uplifts
end

"""
    total_uplift(instance, prices, market_schedule)

Return aggregate make-whole payment.
"""
function total_uplift(instance, prices, market_schedule)
    return sum(unit_uplifts(instance, prices, market_schedule))
end

"""
    settlement_report(instance, prices, market_schedule)

Return one row per generator with production, cost, revenue, profit,
self-schedule profit, and uplift.
"""
function settlement_report(instance, prices, market_schedule)
    components = _settlement_components(instance, prices, market_schedule)

    return DataFrame(
        unit = collect(1:length(components.costs)),
        production = vec(sum(components.production; dims = 2)),
        cost = components.costs,
        revenue = components.revenues,
        profit = components.profits,
        self_schedule_profit = components.self_profits,
        uplift = components.uplifts,
    )
end

function _run_ch_method(
    method::Function,
    instance,
    initial_prices,
    stop;
    verbose::Int = -1,
    kwargs...,
)
    return method(instance, initial_prices, stop; verbose, kwargs...)
end

function _run_ch_method(
    method,
    instance,
    initial_prices,
    stop;
    eps::Float64,
    alpha::Float64,
    smoothing_parameter,
    stepsize::Float64,
    pool_size::Int,
    initial_distance_estimate::Float64,
    obj_sol,
    verbose::Int,
    kwargs...,
)
    name = _normalize_method(method)
    if name in (:column_generation, :cg)
        return ColumnGeneration(instance, initial_prices, stop, eps; pool_size, verbose, kwargs...)
    elseif name in (:bundle_level, :blm)
        return BundleLevelMethod(instance, initial_prices, stop, alpha; verbose, kwargs...)
    elseif name in (:multicut_bundle_level, :mblm)
        return MulticutBundleLevelMethod(instance, initial_prices, stop, alpha; verbose, kwargs...)
    elseif name in (:bundle_proximal, :bundle_proximal_level, :bplm)
        return BundleProximalLevelMethod(
            instance,
            initial_prices,
            stop,
            alpha;
            smoothing_parameter,
            verbose,
            kwargs...,
        )
    elseif name in (:multicut_bundle_proximal, :multicut_bundle_proximal_level, :mbplm)
        return MulticutBundleProximalLevelMethod(
            instance,
            initial_prices,
            stop,
            alpha;
            verbose,
            kwargs...,
        )
    elseif name in (:preconditioned_level, :preconditioned_bundle_level, :pc_blm)
        return PreconditionedLevelMethod(instance, initial_prices, stop, alpha; verbose, kwargs...)
    elseif name in (:preconditioned_proximal, :preconditioned_bundle_proximal, :pc_bplm)
        return PreconditionedProximalLevelMethod(
            instance,
            initial_prices,
            stop,
            alpha;
            verbose,
            kwargs...,
        )
    elseif name in (:dynamic_level, :dlm)
        return DynamicLevelMethod(instance, initial_prices, stop, alpha; verbose, kwargs...)
    elseif name in (:subgradient, :subg)
        return SubgradientMethod(instance, initial_prices, stop, alpha; verbose, kwargs...)
    elseif name in (:estimated_polyak, :est_polyak)
        return EstimatedPolyak(instance, initial_prices, stop, alpha; verbose, kwargs...)
    elseif name in (:polyak, :polyak_method)
        isnothing(obj_sol) && throw(ArgumentError("method=:polyak requires obj_sol"))
        return PolyakMethod(instance, initial_prices, stop, obj_sol; verbose, kwargs...)
    elseif name in (:d_adaptation, :dadaptation, :da)
        return DAdaptation(
            instance,
            initial_prices,
            stop,
            initial_distance_estimate;
            smoothing_parameter,
            verbose,
            kwargs...,
        )
    elseif name in (:dowg, :dog)
        return DowG(
            instance,
            initial_prices,
            stop,
            initial_distance_estimate;
            smoothing_parameter,
            verbose,
            kwargs...,
        )
    elseif name in (:fast_gradient, :fgm)
        return FastGradientMethod(
            instance,
            initial_prices,
            stop,
            smoothing_parameter;
            stepsize,
            verbose,
            kwargs...,
        )
    else
        throw(ArgumentError("unsupported convex-hull pricing method: $method"))
    end
end

_budget_to_stop(budget::StoppingCriterion) = budget
_budget_to_stop(budget::Integer) = IterationLimit(Int(budget))
_budget_to_stop(budget::Real) = TimeBudget(Float64(budget))
_budget_to_stop(budget) =
    throw(ArgumentError("budget must be a StoppingCriterion, integer, or real number"))

_normalize_method(method::Symbol) = Symbol(replace(lowercase(String(method)), "-" => "_"))
_normalize_method(method::AbstractString) = _normalize_method(Symbol(method))
_normalize_method(method) = throw(ArgumentError("method must be a Symbol, string, or callable"))

function _price_vector(p, prices)
    price_vec = Float64.(collect(prices))
    length(price_vec) == p.T ||
        throw(DimensionMismatch("expected $(p.T) prices, got $(length(price_vec))"))
    return price_vec
end

function _require_values(model, context)
    has_values(model) && return nothing
    throw(ErrorException("$context did not produce a solution; status = $(termination_status(model))"))
end

function _extract_gen_time_values(var, nb_gen::Int, horizon::Int)
    return [value(var[gen, t]) for gen = 1:nb_gen, t = 1:horizon]
end

function _unit_production_costs(p, production, commitment, startup)
    costs = zeros(Float64, p.NbGen)
    for gen = 1:p.NbGen
        for t = 1:p.T
            costs[gen] +=
                p.NoLoadConsumption[gen] * commitment[gen, t] +
                p.FixedCost[gen] * startup[gen, t] +
                p.MarginalCost[gen] * production[gen, t]
        end
    end
    return costs
end

function _unit_revenues(prices, production)
    nb_gen, horizon = size(production)
    return [sum(prices[t] * production[gen, t] for t = 1:horizon) for gen = 1:nb_gen]
end

function _settlement_components(instance, prices, market_schedule)
    p = unpack_instance(instance)
    price_vec = _price_vector(p, prices)
    production = Float64.(market_schedule.production)
    commitment = Float64.(market_schedule.commitment)
    startup = Float64.(market_schedule.startup)
    _check_schedule_dimensions(p, production, commitment, startup)

    costs = _unit_production_costs(p, production, commitment, startup)
    revenues = _unit_revenues(price_vec, production)
    profits = revenues .- costs
    self = self_schedule(instance, price_vec)
    uplifts = max.(0.0, self.profits .- profits)

    return (
        production = production,
        costs = costs,
        revenues = revenues,
        profits = profits,
        self_profits = self.profits,
        uplifts = uplifts,
    )
end

function _check_schedule_dimensions(p, production, commitment, startup)
    expected = (p.NbGen, p.T)
    size(production) == expected ||
        throw(DimensionMismatch("production has size $(size(production)), expected $expected"))
    size(commitment) == expected ||
        throw(DimensionMismatch("commitment has size $(size(commitment)), expected $expected"))
    size(startup) == expected ||
        throw(DimensionMismatch("startup has size $(size(startup)), expected $expected"))
    return nothing
end
