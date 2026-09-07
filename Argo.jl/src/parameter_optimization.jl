module ParameterOptimization

using ..Argo

export OptimizationQuery,
    OptimizationResult,
    RegisteredStrategy,
    ScalarParameter,
    StrategyRegistry,
    add_strategy,
    declared_parameters,
    default_strategies,
    explore,
    isoptimized,
    moreau,
    optimize

"""One bounded scalar parameter admitted by a certificate or reformulation."""
struct ScalarParameter
    name::Symbol
    lower::Float64
    upper::Float64
    lower_closed::Bool
    upper_closed::Bool
    scale::Symbol

    function ScalarParameter(
        name::Symbol,
        lower::Real,
        upper::Real;
        lower_closed::Bool=true,
        upper_closed::Bool=true,
        scale::Symbol=:linear,
    )
        isfinite(lower) && isfinite(upper) && lower <= upper ||
            throw(ArgumentError("parameter endpoints must be finite and ordered"))
        lower == upper && !(lower_closed && upper_closed) && throw(
            ArgumentError("an open parameter interval cannot have equal endpoints"),
        )
        scale in (:linear, :log) ||
            throw(ArgumentError("parameter scale must be :linear or :log"))
        scale === :log &&
            lower <= 0 &&
            throw(ArgumentError("a log-scaled parameter must be positive"))
        return new(
            name,
            Float64(lower),
            Float64(upper),
            lower_closed,
            upper_closed,
            scale,
        )
    end
end

function _effective_lower(parameter::ScalarParameter)
    return parameter.lower_closed ? parameter.lower : nextfloat(parameter.lower)
end

function _effective_upper(parameter::ScalarParameter)
    return parameter.upper_closed ? parameter.upper : prevfloat(parameter.upper)
end

function _coordinate(parameter::ScalarParameter, fraction::Real)
    numeric_lower = _effective_lower(parameter)
    numeric_upper = _effective_upper(parameter)
    if parameter.scale === :linear
        return numeric_lower + fraction * (numeric_upper - numeric_lower)
    end
    lower = log(numeric_lower)
    upper = log(numeric_upper)
    return exp(lower + fraction * (upper - lower))
end


"""The complete input seen by a provenance-bearing optimization strategy."""
struct OptimizationQuery
    certificates::Vector{Argo.Certificate}
    parameters::Vector{ScalarParameter}
    request::Union{Argo.RankingRequest,Argo.FixedBudgetRequest}
end

"""A justified optimizer registered for one explicitly checked problem class."""
struct RegisteredStrategy
    id::Symbol
    provenance::String
    supports::Function
    candidates::Function

    function RegisteredStrategy(
        id::Symbol,
        provenance::AbstractString,
        supports::Function,
        candidates::Function,
    )
        isempty(provenance) &&
            throw(ArgumentError("an optimization strategy needs scientific provenance"))
        return new(id, String(provenance), supports, candidates)
    end
end

"""An ordered collection of justified parameter-optimization strategies."""
struct StrategyRegistry
    strategies::Vector{RegisteredStrategy}

    function StrategyRegistry(strategies::Vector{RegisteredStrategy})
        ids = getfield.(strategies, :id)
        length(ids) == length(unique(ids)) ||
            throw(ArgumentError("duplicate parameter-optimization strategy id"))
        return new(strategies)
    end
end

StrategyRegistry(strategies::RegisteredStrategy...) = StrategyRegistry(collect(strategies))
StrategyRegistry() = StrategyRegistry(RegisteredStrategy[])

function add_strategy(registry::StrategyRegistry, strategies::RegisteredStrategy...)
    return StrategyRegistry(vcat(registry.strategies, collect(strategies)))
end

"""The explicit success or non-applicability result of justified optimization."""
struct OptimizationResult
    status::Symbol
    strategy::Union{Nothing,Symbol}
    assessment::Union{Nothing,Argo.PlanAssessment}
    assignments::Dict{Symbol,Argo.ScalarExpr}
    reason::String

    function OptimizationResult(
        status::Symbol,
        strategy::Union{Nothing,Symbol},
        assessment::Union{Nothing,Argo.PlanAssessment},
        assignments::Dict{Symbol,Argo.ScalarExpr},
        reason::AbstractString,
    )
        status in (:optimized, :not_optimized) ||
            throw(ArgumentError("unknown optimization status $status"))
        status === :optimized && (strategy === nothing || assessment === nothing) &&
            throw(ArgumentError("an optimized result needs a strategy and assessment"))
        status === :not_optimized && (strategy !== nothing || assessment !== nothing) &&
            throw(ArgumentError("a non-optimized result cannot contain an assessment"))
        return new(status, strategy, assessment, assignments, String(reason))
    end
end

isoptimized(result::OptimizationResult) = result.status === :optimized

const DEFAULT_STRATEGIES = StrategyRegistry()
default_strategies() = deepcopy(DEFAULT_STRATEGIES)

function _values_with_parameter(values, parameter::ScalarParameter, value::Real)
    result = Dict{Any,Any}(pairs(values))
    result[parameter.name] = value
    return result
end

function _evaluate(certificates, parameter, value, request)
    parameter_values = _values_with_parameter(request.context.values, parameter, value)
    ranked = Argo.rank(certificates, Argo.with(request; values=parameter_values))
    return isempty(ranked) ? nothing : first(ranked)
end

function _record_parameter(assessment, parameter, value)
    assignments = copy(assessment.assignments)
    assignments[parameter.name] = Argo.scalar(value)
    return Argo.PlanAssessment(
        assessment.phases, assessment.complexity, assessment.metrics, assignments
    )
end

"""Resolve every theorem-declared free parameter domain for one certificate."""
function declared_parameters(
    certificate::Argo.Certificate;
    k::Integer=1,
    initial_bounds=Dict(),
    values=Dict(),
)
    result = ScalarParameter[]
    names = Symbol[
        plan.name for plan in Argo.Authoring.parameter_plans(certificate) if
        plan.domain !== nothing
    ]
    for step in Argo.Authoring.reformulation_path(certificate)
        append!(names, keys(step.parameter_domains))
    end
    for name in sort!(unique(names); by=string)
        interval = Argo.parameter_interval(
            certificate, name; k=k, initial_bounds=initial_bounds, values=values
        )
        lower = Argo.try_evaluate_scalar(interval.lower, values)
        upper = Argo.try_evaluate_scalar(interval.upper, values)
        lower === nothing && throw(
            ArgumentError("the lower endpoint for parameter $name remains symbolic"),
        )
        upper === nothing && throw(
            ArgumentError("the upper endpoint for parameter $name remains symbolic"),
        )
        push!(
            result,
            ScalarParameter(
                name,
                lower,
                upper;
                lower_closed=interval.lower_closed,
                upper_closed=interval.upper_closed,
                scale=interval.scale,
            ),
        )
    end
    sort!(result; by=parameter -> string(parameter.name))
    return result
end

function _parameter_admits(parameter::ScalarParameter, value::Real)
    isfinite(value) || return false
    lower_ok = parameter.lower_closed ? value >= parameter.lower : value > parameter.lower
    upper_ok = parameter.upper_closed ? value <= parameter.upper : value < parameter.upper
    return lower_ok && upper_ok
end

function _candidate_dict(candidate)
    candidate isa AbstractDict || candidate isa NamedTuple || throw(
        ArgumentError("an optimization strategy candidate must be a dictionary or named tuple"),
    )
    return Dict{Symbol,Float64}(
        begin
            name isa Symbol ||
                throw(ArgumentError("optimization assignment names must be symbols"))
            value isa Real || throw(
                ArgumentError("optimization assignment $name must be a real number"),
            )
            name => Float64(value)
        end for (name, value) in pairs(candidate)
    )
end

function _strategy_candidates(strategy::RegisteredStrategy, query::OptimizationQuery)
    raw = strategy.candidates(query)
    if raw isa AbstractDict || raw isa NamedTuple
        return Dict{Symbol,Float64}[_candidate_dict(raw)]
    elseif raw isa AbstractVector || raw isa Tuple
        return Dict{Symbol,Float64}[_candidate_dict(candidate) for candidate in raw]
    end
    throw(
        ArgumentError(
            "strategy $(strategy.id) must return one assignment or a collection of assignments",
        ),
    )
end

function _validate_candidate!(candidate, parameters, strategy)
    expected = Set(parameter.name for parameter in parameters)
    Set(keys(candidate)) == expected || throw(
        ArgumentError(
            "strategy $(strategy.id) must assign exactly $(join(sort!(string.(collect(expected))), ", "))",
        ),
    )
    for parameter in parameters
        value = candidate[parameter.name]
        _parameter_admits(parameter, value) || throw(
            DomainError(
                value,
                "strategy $(strategy.id) returned an inadmissible value for $(parameter.name)",
            ),
        )
    end
    return candidate
end

function _request_with_assignments(
    request::Union{Argo.RankingRequest,Argo.FixedBudgetRequest}, assignments
)
    values = Dict{Any,Any}(pairs(request.context.values))
    merge!(values, assignments)
    return Argo.with(request; values=values)
end

"""
Jointly optimize declared or explicit parameters only through registered,
provenance-bearing strategies whose preconditions hold. A non-applicable
registry returns an explicit `:not_optimized` result.
"""
function optimize(
    certificates::AbstractVector{<:Argo.Certificate},
    parameters::AbstractVector{<:ScalarParameter},
    request::Union{Argo.RankingRequest,Argo.FixedBudgetRequest};
    strategies::StrategyRegistry=default_strategies(),
)
    isempty(certificates) && return OptimizationResult(
        :not_optimized,
        nothing,
        nothing,
        Dict{Symbol,Argo.ScalarExpr}(),
        "no certificates were supplied",
    )
    isempty(parameters) && throw(ArgumentError("at least one parameter is required"))
    names = getfield.(parameters, :name)
    length(names) == length(unique(names)) ||
        throw(ArgumentError("parameter names must be unique"))
    for name in names
        any(
            certificate -> Argo.Authoring.uses_scalar(certificate, name), certificates
        ) || throw(ArgumentError("no supplied certificate depends on parameter $name"))
    end
    query = OptimizationQuery(
        Argo.Certificate[certificates...], ScalarParameter[parameters...], request
    )
    best = nothing
    best_strategy = nothing
    supported = Symbol[]
    for strategy in strategies.strategies
        applicable = strategy.supports(query)
        applicable isa Bool || throw(
            ArgumentError("strategy $(strategy.id) precondition must return a boolean"),
        )
        applicable || continue
        push!(supported, strategy.id)
        candidates = _strategy_candidates(strategy, query)
        isempty(candidates) && throw(
            ArgumentError("applicable strategy $(strategy.id) returned no candidates"),
        )
        for candidate in candidates
            _validate_candidate!(candidate, parameters, strategy)
            ranked = Argo.rank(
                certificates, _request_with_assignments(request, candidate)
            )
            isempty(ranked) && continue
            assessment = first(ranked)
            metric_name = request isa Argo.RankingRequest ? :oracle_cost : :bound
            if best === nothing ||
               Argo.metric(assessment, metric_name).value <
               Argo.metric(best, metric_name).value
                best = assessment
                best_strategy = strategy.id
            end
        end
    end
    if best === nothing
        reason = isempty(supported) ?
                 "no registered strategy satisfied its preconditions" :
                 "applicable strategies produced no numerically comparable assessment"
        return OptimizationResult(
            :not_optimized,
            nothing,
            nothing,
            Dict{Symbol,Argo.ScalarExpr}(),
            reason,
        )
    end
    assignments = Dict{Symbol,Argo.ScalarExpr}(
        name => best.assignments[name] for name in names
    )
    return OptimizationResult(:optimized, best_strategy, best, assignments, "")
end

function optimize(
    certificates::AbstractVector{<:Argo.Certificate},
    parameters::AbstractVector{<:ScalarParameter};
    accuracy::Union{Nothing,Real}=nothing,
    iterations::Union{Nothing,Integer}=nothing,
    initial_bounds=Dict(),
    oracle_costs=Dict(),
    values=Dict(),
    max_iterations::Integer=1_000_000_000,
    strategies::StrategyRegistry=default_strategies(),
)
    (accuracy === nothing) == (iterations === nothing) && throw(
        ArgumentError("provide exactly one of accuracy or iterations"),
    )
    request = if accuracy === nothing
        Argo.FixedBudgetRequest(;
            iterations=iterations,
            initial_bounds=initial_bounds,
            oracle_costs=oracle_costs,
            values=values,
        )
    else
        Argo.RankingRequest(;
            accuracy=accuracy,
            initial_bounds=initial_bounds,
            oracle_costs=oracle_costs,
            values=values,
            max_iterations=max_iterations,
        )
    end
    return optimize(certificates, parameters, request; strategies=strategies)
end

"""Heuristically search one symbolic scalar by grid and local refinement."""
function explore(
    certificates::AbstractVector{<:Argo.Certificate},
    parameter::ScalarParameter,
    request::Argo.RankingRequest;
    grid_points::Integer=33,
    refinement_steps::Integer=12,
)
    grid_points >= 2 || throw(ArgumentError("grid_points must be at least two"))
    refinement_steps >= 0 || throw(ArgumentError("refinement_steps must be nonnegative"))
    isempty(certificates) && return nothing
    eligible = Argo.Certificate[
        certificate for certificate in certificates if
        Argo.Authoring.uses_scalar(certificate, parameter.name)
    ]
    isempty(eligible) && throw(
        ArgumentError(
            "parameter $(parameter.name) is fixed by the supplied certificates and " *
            "cannot be optimized",
        ),
    )
    cache = Dict{Float64,Union{Nothing,Argo.PlanAssessment}}()
    function evaluate_at(value)
        point = clamp(Float64(value), _effective_lower(parameter), _effective_upper(parameter))
        return get!(cache, point) do
            _evaluate(eligible, parameter, point, request)
        end
    end

    grid = Float64[
        _coordinate(parameter, (index - 1) / (grid_points - 1)) for
        index in 1:Int(grid_points)
    ]
    for point in grid
        evaluate_at(point)
    end
    finite_points = filter(point -> cache[point] !== nothing, grid)
    isempty(finite_points) && return nothing
    best_point = argmin(
        point -> Argo.metric(cache[point], :oracle_cost).value, finite_points
    )
    best_index = findfirst(==(best_point), grid)
    left = grid[max(1, best_index - 1)]
    right = grid[min(length(grid), best_index + 1)]

    inverse_phi = (sqrt(5.0) - 1.0) / 2
    for _ in 1:Int(refinement_steps)
        right > left || break
        c = right - inverse_phi * (right - left)
        d = left + inverse_phi * (right - left)
        c_result = evaluate_at(c)
        d_result = evaluate_at(d)
        c_cost = c_result === nothing ? Inf : Argo.metric(c_result, :oracle_cost).value
        d_cost = d_result === nothing ? Inf : Argo.metric(d_result, :oracle_cost).value
        if c_cost <= d_cost
            right = d
        else
            left = c
        end
    end
    evaluated_points = filter(point -> cache[point] !== nothing, collect(keys(cache)))
    isempty(evaluated_points) && return nothing
    best_evaluation = argmin(
        point -> Argo.metric(cache[point], :oracle_cost).value, evaluated_points
    )
    return _record_parameter(cache[best_evaluation], parameter, best_evaluation)
end

function explore(
    certificates::AbstractVector{<:Argo.Certificate},
    parameter::ScalarParameter;
    accuracy::Real,
    initial_bounds=Dict(),
    oracle_costs=Dict(),
    values=Dict(),
    grid_points::Integer=33,
    refinement_steps::Integer=12,
    max_iterations::Integer=1_000_000_000,
)
    request = Argo.RankingRequest(;
        accuracy=accuracy,
        initial_bounds=initial_bounds,
        oracle_costs=oracle_costs,
        values=values,
        max_iterations=max_iterations,
    )
    return explore(
        certificates,
        parameter,
        request;
        grid_points=grid_points,
        refinement_steps=refinement_steps,
    )
end


# Compatibility for the pre-domain scalar search API. New scientific code
# should call `explore` for this heuristic or the vector `optimize` method for a
# registered, justified strategy.
function optimize(
    certificates::AbstractVector{<:Argo.Certificate},
    parameter::ScalarParameter,
    request::Argo.RankingRequest;
    kwargs...,
)
    return explore(certificates, parameter, request; kwargs...)
end

function optimize(
    certificates::AbstractVector{<:Argo.Certificate},
    parameter::ScalarParameter;
    kwargs...,
)
    return explore(certificates, parameter; kwargs...)
end

function _rotaru_optimal_normalized_step(k::Integer)
    k >= 0 || throw(DomainError(k, "iteration count must be nonnegative"))
    k == 0 && return 1.0
    left = 1.0
    right = 2.0
    for _ in 1:80
        middle = left + (right - left) / 2
        (middle == left || middle == right) && break
        log_growth = -2k * log(middle - 1)
        log_linear = log1p(2k * middle)
        if log_growth > log_linear
            left = middle
        else
            right = middle
        end
    end
    return left + (right - left) / 2
end

function _rotaru_theorem_interval(query::OptimizationQuery)
    length(query.certificates) == 1 || return nothing
    length(query.parameters) == 1 || return nothing
    certificate = only(query.certificates)
    certificate.theorem.id === :rotaru2026_exact_fixed_step_gradient_norm ||
        return nothing
    parameter = only(query.parameters)
    parameter.name === :step_size || return nothing
    interval = try
        Argo.parameter_interval(
            certificate,
            :step_size;
            initial_bounds=query.request.context.initial_bounds,
            values=query.request.context.values,
        )
    catch
        return nothing
    end
    lower = Argo.try_evaluate_scalar(interval.lower, query.request.context.values)
    upper = Argo.try_evaluate_scalar(interval.upper, query.request.context.values)
    lower === nothing && return nothing
    upper === nothing && return nothing
    lower == parameter.lower || return nothing
    upper == parameter.upper || return nothing
    interval.lower_closed == parameter.lower_closed || return nothing
    interval.upper_closed == parameter.upper_closed || return nothing
    return Float64(lower), Float64(upper)
end

function _rotaru_context(query::OptimizationQuery, step_size::Real)
    return _request_with_assignments(
        query.request, Dict{Symbol,Float64}(:step_size => Float64(step_size))
    ).context
end

function _rotaru_evaluation(query::OptimizationQuery, k::Integer, upper::Float64)
    normalized_step = _rotaru_optimal_normalized_step(k)
    step_size = normalized_step * upper / 2
    evaluation = Argo.Authoring.evaluate_bound(
        only(query.certificates), k, _rotaru_context(query, step_size)
    )
    return evaluation, step_size
end

function _rotaru_supports(query::OptimizationQuery)
    endpoints = _rotaru_theorem_interval(query)
    endpoints === nothing && return false
    lower, upper = endpoints
    lower == 0 && upper > 0 || return false
    evaluation, _ = _rotaru_evaluation(query, 0, upper)
    return evaluation.status === Argo.Authoring.NumericBound
end

function _rotaru_candidate(query::OptimizationQuery)
    _, upper_endpoint = something(_rotaru_theorem_interval(query))
    if query.request isa Argo.FixedBudgetRequest
        _, step_size = _rotaru_evaluation(
            query, query.request.iterations, upper_endpoint
        )
        return Dict(:step_size => step_size)
    end
    target = Float64(query.request.accuracy)
    limit = query.request.context.max_iterations
    initial, initial_step = _rotaru_evaluation(query, 0, upper_endpoint)
    initial.value <= target && return Dict(:step_size => initial_step)
    lower = 0
    upper = min(1, limit)
    successful = false
    while upper <= limit
        evaluation, _ = _rotaru_evaluation(query, upper, upper_endpoint)
        if evaluation.status === Argo.Authoring.NumericBound && evaluation.value <= target
            successful = true
            break
        end
        lower = upper
        upper == limit && break
        upper = min(limit, max(upper + 1, 2 * upper))
    end
    if successful
        while lower + 1 < upper
            middle = lower + (upper - lower) ÷ 2
            evaluation, _ = _rotaru_evaluation(query, middle, upper_endpoint)
            if evaluation.status === Argo.Authoring.NumericBound &&
               evaluation.value <= target
                upper = middle
            else
                lower = middle
            end
        end
    else
        upper = limit
    end
    _, step_size = _rotaru_evaluation(query, upper, upper_endpoint)
    return Dict(:step_size => step_size)
end

const ROTARU_FIXED_STEP_STRATEGY = RegisteredStrategy(
    :rotaru2026_exact_fixed_step,
    "Rotaru 2026 exact bound: the minimum is at the unique crossing of the increasing linear and decreasing post-unit branches",
    _rotaru_supports,
    _rotaru_candidate,
)
push!(DEFAULT_STRATEGIES.strategies, ROTARU_FIXED_STEP_STRATEGY)

"""Optimize the one Moreau-envelope parameter without enabling a parameter grid."""
function moreau(
    problem::Argo.Problem,
    parameter::ScalarParameter,
    ranking::Argo.RankingRequest,
    applicability::Argo.ApplicabilityRequest=Argo.ApplicabilityRequest();
    kwargs...,
)
    parameter.name === :lambda ||
        throw(ArgumentError("the Moreau-envelope parameter must be named :lambda"))
    symbolic_lambda = Argo.symbolic(:lambda; scope=:moreau_envelope)
    request = Argo.with(
        applicability;
        reformulations=Argo.Authoring.AbstractReformulationRule[Argo.Authoring.MoreauEnvelopeRule(
            symbolic_lambda
        ),],
        reformulation_depth=1,
    )
    discovered = Argo.certificates(problem, request)
    reformulated = Argo.Certificate[
        certificate for certificate in discovered if
        Argo.Authoring.has_reformulation(certificate, :moreau_envelope)
    ]
    return explore(reformulated, parameter, ranking; kwargs...)
end

function moreau(
    problem::Argo.Problem,
    parameter::ScalarParameter;
    quantity::Symbol=:objective_gap,
    accuracy::Real,
    initial_bounds=Dict(),
    oracle_costs=Dict(),
    values=Dict(),
    catalogue::Argo.Authoring.Catalogue=Argo.Authoring.default_catalogue(),
    resolver::Argo.Authoring.AbstractOracleResolver=Argo.Authoring.DirectOracleResolver(),
    rules::Argo.Authoring.RuleSet=Argo.Authoring.default_rules(),
    max_bindings::Integer=4096,
    max_iterations::Integer=1_000_000_000,
    kwargs...,
)
    ranking = Argo.RankingRequest(;
        accuracy=accuracy,
        initial_bounds=initial_bounds,
        oracle_costs=oracle_costs,
        values=values,
        max_iterations=max_iterations,
    )
    applicability = Argo.ApplicabilityRequest(;
        quantity=quantity,
        catalogue=catalogue,
        resolver=resolver,
        rules=rules,
        max_bindings=max_bindings,
    )
    return moreau(problem, parameter, ranking, applicability; kwargs...)
end

end
