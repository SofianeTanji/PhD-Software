import LinearAlgebra: rank

"""Numerical assumptions shared by bound evaluation and oracle-cost comparison."""
struct ComparisonContext
    initial_bounds::Dict{Any,Any}
    oracle_costs::Dict{Any,Any}
    values::Dict{Any,Any}
    max_iterations::Int
end

function ComparisonContext(;
    initial_bounds=Dict(),
    oracle_costs=Dict(),
    values=Dict(),
    max_iterations::Integer=1_000_000_000,
)
    max_iterations >= 0 || throw(ArgumentError("max_iterations must be nonnegative"))
    return ComparisonContext(
        Dict{Any,Any}(pairs(initial_bounds)),
        Dict{Any,Any}(pairs(oracle_costs)),
        Dict{Any,Any}(pairs(values)),
        Int(max_iterations),
    )
end

"""A stable request to rank applicable certificates at a numerical accuracy."""
struct RankingRequest
    accuracy::Real
    context::ComparisonContext

    function RankingRequest(accuracy::Real, context::ComparisonContext)
        isfinite(accuracy) && accuracy >= 0 ||
            throw(ArgumentError("accuracy must be finite and nonnegative"))
        return new(accuracy, context)
    end
end

"""A stable request to compare certificate bounds at one fixed iteration budget."""
struct FixedBudgetRequest
    iterations::Int
    context::ComparisonContext

    function FixedBudgetRequest(iterations::Integer, context::ComparisonContext)
        iterations >= 0 || throw(ArgumentError("iterations must be nonnegative"))
        return new(Int(iterations), context)
    end
end

function FixedBudgetRequest(;
    iterations::Integer,
    initial_bounds=Dict(),
    oracle_costs=Dict(),
    values=Dict(),
)
    return FixedBudgetRequest(
        iterations,
        ComparisonContext(;
            initial_bounds=initial_bounds,
            oracle_costs=oracle_costs,
            values=values,
            max_iterations=iterations,
        ),
    )
end

function RankingRequest(;
    accuracy::Real,
    initial_bounds=Dict(),
    oracle_costs=Dict(),
    values=Dict(),
    max_iterations::Integer=1_000_000_000,
)
    return RankingRequest(
        accuracy,
        ComparisonContext(;
            initial_bounds=initial_bounds,
            oracle_costs=oracle_costs,
            values=values,
            max_iterations=max_iterations,
        ),
    )
end

function with(
    context::ComparisonContext;
    initial_bounds=context.initial_bounds,
    oracle_costs=context.oracle_costs,
    values=context.values,
    max_iterations=context.max_iterations,
)
    return ComparisonContext(;
        initial_bounds=initial_bounds,
        oracle_costs=oracle_costs,
        values=values,
        max_iterations=max_iterations,
    )
end

function with(
    request::FixedBudgetRequest;
    iterations=request.iterations,
    initial_bounds=request.context.initial_bounds,
    oracle_costs=request.context.oracle_costs,
    values=request.context.values,
)
    return FixedBudgetRequest(;
        iterations=iterations,
        initial_bounds=initial_bounds,
        oracle_costs=oracle_costs,
        values=values,
    )
end

function with(
    request::RankingRequest;
    accuracy=request.accuracy,
    initial_bounds=request.context.initial_bounds,
    oracle_costs=request.context.oracle_costs,
    values=request.context.values,
    max_iterations=request.context.max_iterations,
)
    return RankingRequest(;
        accuracy=accuracy,
        initial_bounds=initial_bounds,
        oracle_costs=oracle_costs,
        values=values,
        max_iterations=max_iterations,
    )
end

"""A composed call's role provenance and the source role used for costing."""
struct OracleRolePath
    segments::Tuple{Vararg{Symbol}}
    cost_role::Symbol

    function OracleRolePath(segments::Tuple{Vararg{Symbol}}, cost_role::Symbol)
        isempty(segments) && throw(ArgumentError("an oracle role path must not be empty"))
        cost_role in segments ||
            throw(ArgumentError("the oracle cost role must occur in its role path"))
        return new(segments, cost_role)
    end
end

OracleRolePath(role::Symbol) = OracleRolePath((role,), role)

function _role_label(path::OracleRolePath)
    return Symbol(join(string.(path.segments), "__"))
end

"""The role-specific oracle calls required to realize a certificate."""
struct OracleWork
    role_path::OracleRolePath
    oracle::Symbol
    count::Int
    declared_cost::Union{Nothing,ScalarExpr}
end

function OracleWork(
    role::Symbol, oracle::Symbol, count::Integer, declared_cost::Union{Nothing,ScalarExpr}
)
    return OracleWork(OracleRolePath(role), oracle, Int(count), declared_cost)
end

function Base.getproperty(work::OracleWork, name::Symbol)
    name === :role && return _role_label(getfield(work, :role_path))
    return getfield(work, name)
end

function Base.propertynames(::OracleWork, private::Bool=false)
    fields = (:role, :oracle, :count, :declared_cost)
    return private ? (:role_path, fields...) : fields
end

oracle_role_path(work::OracleWork) = work.role_path

function prefix_oracle_work(
    work::OracleWork,
    prefix::Symbol;
    cost_role::Symbol=work.role_path.cost_role,
    multiplier::Integer=1,
)
    multiplier >= 0 || throw(ArgumentError("an oracle-work multiplier must be nonnegative"))
    path = OracleRolePath((prefix, work.role_path.segments...), cost_role)
    return OracleWork(path, work.oracle, Int(multiplier) * work.count, work.declared_cost)
end

struct OracleComplexity
    iterations::Int
    calls::Vector{OracleWork}
end

"""One certificate-backed phase and its chosen iteration count, when known."""
struct PlanPhase
    certificate::Certificate
    iterations::Union{Nothing,Int}
end

"""One named numerical assessment, optionally with a standard error."""
struct MetricEstimate
    name::Symbol
    value::Float64
    standard_error::Union{Nothing,Float64}
end

"""A uniform assessment returned by core ranking and every ranking extension."""
struct PlanAssessment
    phases::Vector{PlanPhase}
    complexity::Union{Nothing,OracleComplexity}
    metrics::Vector{MetricEstimate}
    assignments::Dict{Symbol,ScalarExpr}
end

function metric(assessment::PlanAssessment, name::Symbol)
    index = findfirst(value -> value.name === name, assessment.metrics)
    index === nothing && throw(KeyError(name))
    return assessment.metrics[index]
end

function _scalar_assignments(values)
    result = Dict{Symbol,ScalarExpr}()
    for (name, value) in pairs(values)
        name isa Symbol || continue
        value isa Real || value isa ScalarExpr || continue
        result[name] = scalar(value)
    end
    return result
end

@enum BoundEvaluationStatus::UInt8 begin
    NumericBound
    SymbolicBound
    InvalidBound
end

"""The typed outcome of evaluating a certificate bound numerically."""
struct BoundEvaluation
    status::BoundEvaluationStatus
    value::Union{Nothing,Float64}
end

function evaluate_bound(
    certificate::Certificate, k::Integer, context::ComparisonContext=ComparisonContext()
)
    expression = try
        certificate_bound(
            certificate, k; initial_bounds=context.initial_bounds, values=context.values
        )
    catch error
        error isa DomainError && return BoundEvaluation(InvalidBound, nothing)
        rethrow()
    end
    value = try_evaluate_scalar(expression, context.values)
    value === nothing && return BoundEvaluation(SymbolicBound, nothing)
    value isa Real || return BoundEvaluation(SymbolicBound, nothing)
    numeric = Float64(value)
    isfinite(numeric) && numeric >= 0 || return BoundEvaluation(InvalidBound, nothing)
    return BoundEvaluation(NumericBound, numeric)
end

"""
    iterations_to_accuracy(certificate; accuracy, ...)

Invert a monotone certificate bound using generic exponential and integer binary
search. Return `nothing` when the bound remains symbolic or the budget is not
reached within `max_iterations`.
"""
function iterations_to_accuracy(certificate::Certificate, request::RankingRequest)
    target = Float64(request.accuracy)
    context = request.context

    evaluated = evaluate_bound(certificate, 0, context)
    evaluated.status === SymbolicBound && return nothing
    evaluated.status === NumericBound && evaluated.value <= target && return 0

    lower = 0
    upper = min(1, context.max_iterations)
    success = false
    while upper <= context.max_iterations
        evaluated = evaluate_bound(certificate, upper, context)
        evaluated.status === SymbolicBound && return nothing
        if evaluated.status === NumericBound && evaluated.value <= target
            success = true
            break
        end
        lower = upper
        upper == context.max_iterations && break
        upper = min(context.max_iterations, max(upper + 1, 2 * upper))
    end
    success || return nothing

    while lower + 1 < upper
        middle = lower + (upper - lower) ÷ 2
        evaluated = evaluate_bound(certificate, middle, context)
        evaluated.status === SymbolicBound && return nothing
        if evaluated.status === NumericBound && evaluated.value <= target
            upper = middle
        else
            lower = middle
        end
    end
    return upper
end

function iterations_to_accuracy(certificate::Certificate; kwargs...)
    iterations_to_accuracy(certificate, RankingRequest(; kwargs...))
end

function expand_oracle_call(
    ::AbstractOracleEvidencePayload,
    evidence::OracleEvidence,
    certificate::Certificate,
    call::OracleCall,
    iterations::Integer,
    ::ComparisonContext,
)
    return OracleWork[OracleWork(
        call.role, call.oracle, call.count * Int(iterations), evidence.available.cost
    ),]
end

function merge_oracle_work(values::Vector{OracleWork})
    result = OracleWork[]
    positions = Dict{Tuple{OracleRolePath,Symbol,Union{Nothing,ScalarExpr}},Int}()
    for value in values
        key = (value.role_path, value.oracle, value.declared_cost)
        index = get(positions, key, 0)
        if index == 0
            push!(result, value)
            positions[key] = length(result)
        else
            previous = result[index]
            result[index] = OracleWork(
                previous.role_path,
                previous.oracle,
                previous.count + value.count,
                previous.declared_cost,
            )
        end
    end
    return result
end

function _moreau_source_work(step::ReformulationStep, values::Vector{OracleWork})
    proximal = get_oracle(step.source, :prox)
    proximal === nothing && return nothing
    function source_call(value::OracleWork, oracle_name::Symbol, available::Oracle)
        return OracleWork(value.role_path, oracle_name, value.count, available.cost)
    end
    result = OracleWork[]
    for value in values
        if value.oracle === :gradient
            push!(result, source_call(value, :prox, proximal))
        elseif value.oracle === :value
            source_value = get_oracle(step.source, :value)
            source_value === nothing && return nothing
            push!(result, source_call(value, :prox, proximal))
            push!(result, source_call(value, :value, source_value))
        else
            push!(result, value)
        end
    end
    if step.output_map.name === :prox
        path = isempty(result) ? OracleRolePath(:objective) : first(result).role_path
        push!(result, OracleWork(path, :prox, 1, proximal.cost))
    end
    return merge_oracle_work(result)
end

function _source_oracle_work(certificate::Certificate, values::Vector{OracleWork})
    result = values
    for step in Iterators.reverse(certificate.reformulation_steps)
        if step.rule === :moreau_envelope
            pulled_back = _moreau_source_work(step, result)
            pulled_back === nothing && return nothing
            result = pulled_back
        end
    end
    return result
end

function oracle_complexity(
    certificate::Certificate, iterations::Integer, context::ComparisonContext
)
    iterations >= 0 || throw(ArgumentError("iterations must be nonnegative"))
    work = OracleWork[]
    for call in certificate.plan.method.calls
        index = findfirst(
            evidence ->
                evidence.role === call.role && evidence.requirement.name === call.oracle,
            certificate.plan.oracle_evidence,
        )
        index === nothing && return nothing
        evidence = certificate.plan.oracle_evidence[index]
        expanded = expand_oracle_call(
            evidence.payload, evidence, certificate, call, iterations, context
        )
        expanded === nothing && return nothing
        append!(work, expanded)
    end
    source_work = _source_oracle_work(certificate, merge_oracle_work(work))
    source_work === nothing && return nothing
    return OracleComplexity(Int(iterations), source_work)
end

function oracle_complexity(certificate::Certificate, iterations::Integer; kwargs...)
    oracle_complexity(certificate, iterations, ComparisonContext(; kwargs...))
end

function _requested_oracle_cost(oracle_costs, call::OracleWork)
    path_key = (call.role_path, call.oracle)
    haskey(oracle_costs, path_key) && return scalar(oracle_costs[path_key])
    role_key = (call.role, call.oracle)
    haskey(oracle_costs, role_key) && return scalar(oracle_costs[role_key])
    source_key = (call.role_path.cost_role, call.oracle)
    haskey(oracle_costs, source_key) && return scalar(oracle_costs[source_key])
    haskey(oracle_costs, call.oracle) && return scalar(oracle_costs[call.oracle])
    return nothing
end

function _call_cost(call::OracleWork, oracle_costs)
    requested = _requested_oracle_cost(oracle_costs, call)
    requested === nothing || return requested
    return call.declared_cost === nothing ? scalar(1) : call.declared_cost
end

function complexity_cost(
    complexity::OracleComplexity, context::ComparisonContext=ComparisonContext()
)
    total = scalar(0)
    for call in complexity.calls
        total += call.count * _call_cost(call, context.oracle_costs)
    end
    return substitute_scalars(total, context.values)
end

"""
    rank(certificates; accuracy, initial_bounds, oracle_costs)

Rank only numerically comparable certificates by the weighted cost of the
role-specific oracle calls needed to reach the requested accuracy.
"""
function rank(certificates::AbstractVector{<:Certificate}, request::RankingRequest)
    result = PlanAssessment[]
    for certificate in certificates
        iterations = iterations_to_accuracy(certificate, request)
        iterations === nothing && continue
        complexity = oracle_complexity(certificate, iterations, request.context)
        complexity === nothing && continue
        expression = complexity_cost(complexity, request.context)
        numeric = try_evaluate_scalar(expression, request.context.values)
        numeric === nothing && continue
        score = Float64(numeric)
        isfinite(score) && score >= 0 || continue
        push!(
            result,
            PlanAssessment(
                PlanPhase[PlanPhase(certificate, iterations)],
                complexity,
                MetricEstimate[MetricEstimate(:oracle_cost, score, nothing)],
                _scalar_assignments(request.context.values),
            ),
        )
    end
    sort!(
        result;
        by=value -> (
            metric(value, :oracle_cost).value,
            only(value.phases).iterations,
            string(method_declaration(only(value.phases).certificate).id),
            string(only(value.phases).certificate.theorem.id),
        ),
    )
    return result
end

"""Rank numerical certificate bounds at a common, fixed iteration budget."""
function rank(certificates::AbstractVector{<:Certificate}, request::FixedBudgetRequest)
    result = PlanAssessment[]
    for certificate in certificates
        evaluated = evaluate_bound(certificate, request.iterations, request.context)
        evaluated.status === NumericBound || continue
        metrics = MetricEstimate[MetricEstimate(:bound, evaluated.value, nothing)]
        complexity = oracle_complexity(certificate, request.iterations, request.context)
        if complexity !== nothing
            expression = complexity_cost(complexity, request.context)
            cost = try_evaluate_scalar(expression, request.context.values)
            cost === nothing ||
                (isfinite(cost) && cost >= 0) &&
                push!(metrics, MetricEstimate(:oracle_cost, Float64(cost), nothing))
        end
        push!(
            result,
            PlanAssessment(
                PlanPhase[PlanPhase(certificate, request.iterations)],
                complexity,
                metrics,
                _scalar_assignments(request.context.values),
            ),
        )
    end
    sort!(
        result;
        by=value -> (
            metric(value, :bound).value,
            string(method_declaration(only(value.phases).certificate).id),
            string(only(value.phases).certificate.theorem.id),
        ),
    )
    return result
end

function rank(certificates::AbstractVector{<:Certificate}; kwargs...)
    rank(certificates, RankingRequest(; kwargs...))
end

function Base.show(io::IO, value::OracleComplexity)
    print(io, "OracleComplexity(", value.iterations, " iterations; ")
    for (index, call) in enumerate(value.calls)
        index > 1 && print(io, ", ")
        print(io, call.role, '.', call.oracle, '=', call.count)
    end
    return print(io, ')')
end

function Base.show(io::IO, value::MetricEstimate)
    print(io, value.name, '=', value.value)
    value.standard_error === nothing || print(io, " ± ", value.standard_error)
    return nothing
end

function Base.show(io::IO, value::PlanAssessment)
    print(io, "PlanAssessment(", length(value.phases), " phase")
    length(value.phases) == 1 || print(io, 's')
    for estimate in value.metrics
        print(io, ", ")
        show(io, estimate)
    end
    return print(io, ')')
end
