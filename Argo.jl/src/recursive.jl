module Recursive

using ..Argo

import ..Argo.Authoring: expand_oracle_call, resolve_oracle

export NestedPlan, Resolver, certificates, proximal_subproblem

"""A recursive oracle resolver with an explicit remaining nesting depth."""
struct Resolver <: Argo.Authoring.AbstractOracleResolver
    depth::Int
    include_direct::Bool

    function Resolver(; depth::Integer=1, include_direct::Bool=true)
        depth >= 0 || throw(ArgumentError("recursive depth must be nonnegative"))
        return new(Int(depth), include_direct)
    end
end

"""The inner certificate that realizes one controlled inexact prox call."""
struct NestedPlan <: Argo.Authoring.AbstractOracleEvidencePayload
    certificate::Argo.Certificate
    accuracy::Symbol
    initial_bound::Symbol
    initial_value::Argo.ScalarExpr
    kappa::Argo.ScalarExpr
    max_iterations::Int
end

function _requirement_scalar(query::Argo.Authoring.OracleQuery, parameter::Symbol)
    return get(query.constants, parameter) do
        Argo.symbolic(parameter; scope=query.theorem.id)
    end
end

"""Build the induced minimization problem defining a proximal operation."""
function proximal_subproblem(value::Argo.Term, kappa::Union{Real,Argo.ScalarExpr})
    kappa isa Real &&
        kappa <= 0 &&
        throw(ArgumentError("proximal curvature must be positive"))
    curvature = Argo.scalar(kappa)
    regularizer = Argo.term(
        :proximal_regularizer;
        properties=Argo.Property[
            Argo.centered_quadratic(curvature),
            Argo.quadratic(; lambda_min=curvature, lambda_max=curvature),
            Argo.convex(),
            Argo.smooth(curvature),
            Argo.strongly_convex(curvature),
        ],
        oracles=Argo.Oracle[
            Argo.oracle(:value; cost=0),
            Argo.oracle(:gradient; cost=0),
            Argo.oracle(:prox; cost=0),
        ],
    )
    return Argo.minimize(value + regularizer)
end

function resolve_oracle(resolver::Resolver, query::Argo.Authoring.OracleQuery)
    result = if resolver.include_direct
        Argo.Authoring.resolve_oracle(Argo.Authoring.DirectOracleResolver(), query)
    else
        Argo.Authoring.OracleEvidence[]
    end
    resolver.depth > 0 || return result
    query.requirement.name === :prox || return result
    query.requirement.exactness === Argo.ControlledInexactOracle || return result
    contract = query.requirement.inexactness
    contract === nothing && return result
    contract.model === :absolute_subproblem_gap || return result

    kappa = _requirement_scalar(query, contract.kappa)
    inner = proximal_subproblem(query.term, kappa)
    nested_resolver = Resolver(;
        depth=(resolver.depth - 1), include_direct=resolver.include_direct
    )
    inner_certificates = Argo.certificates(
        inner;
        quantity=contract.inner.quantity,
        catalogue=query.catalogue,
        resolver=nested_resolver,
        reformulations=Argo.Authoring.AbstractReformulationRule[],
        reformulation_depth=0,
    )
    for inner_certificate in inner_certificates
        payload = NestedPlan(
            inner_certificate,
            contract.accuracy,
            contract.inner.initial_bound,
            contract.inner.initial_value,
            kappa,
            contract.inner.max_iterations,
        )
        push!(
            result,
            Argo.Authoring.OracleEvidence(
                query.requirement,
                Argo.oracle(:prox; exactness=:controlled_inexact),
                query.role,
                :recursive,
                payload,
            ),
        )
    end
    return result
end

function _accuracy_at(plan::NestedPlan, outer::Argo.Certificate, index::Int, context)
    parameter_index = findfirst(
        parameter -> parameter.name === plan.accuracy, Argo.Authoring.parameter_plans(outer)
    )
    expression = if parameter_index !== nothing
        Argo.parameter_value(
            outer,
            plan.accuracy;
            k=index,
            initial_bounds=context.initial_bounds,
            values=context.values,
        )
    elseif haskey(context.values, plan.accuracy)
        raw = context.values[plan.accuracy]
        if raw isa Function
            Argo.scalar(raw(index))
        elseif raw isa AbstractVector || raw isa Tuple
            length(raw) >= index || return nothing
            Argo.scalar(raw[index])
        elseif raw isa Real || raw isa Argo.ScalarExpr
            Argo.scalar(raw)
        else
            return nothing
        end
    else
        return nothing
    end
    value = Argo.try_evaluate_scalar(expression, context.values)
    value isa Real || return nothing
    numeric = Float64(value)
    isfinite(numeric) && numeric >= 0 || return nothing
    return numeric
end

function _prefix_work(prefix::Symbol, work::Argo.Authoring.OracleWork, multiplier::Int)
    return Argo.Authoring.prefix_oracle_work(
        work, prefix; cost_role=prefix, multiplier=multiplier
    )
end

function expand_oracle_call(
    plan::NestedPlan,
    evidence::Argo.Authoring.OracleEvidence,
    outer::Argo.Certificate,
    call::Argo.Authoring.OracleCall,
    iterations::Integer,
    context::Argo.ComparisonContext,
)
    result = Argo.Authoring.OracleWork[]
    inner_context = Argo.with(
        context;
        initial_bounds=Dict(plan.initial_bound => plan.initial_value),
        max_iterations=min(context.max_iterations, plan.max_iterations),
    )
    for index in 1:Int(iterations)
        accuracy = _accuracy_at(plan, outer, index, context)
        accuracy === nothing && return nothing
        inner_iterations = Argo.iterations_to_accuracy(
            plan.certificate, Argo.RankingRequest(accuracy, inner_context)
        )
        inner_iterations === nothing && return nothing
        complexity = Argo.oracle_complexity(
            plan.certificate, inner_iterations, inner_context
        )
        complexity === nothing && return nothing
        for work in complexity.calls
            push!(result, _prefix_work(call.role, work, call.count))
        end
    end
    return result
end

"""Discover certificates with recursive controlled-oracle resolution enabled."""
function certificates(problem::Argo.Problem; depth::Integer=1, kwargs...)
    return Argo.certificates(problem; resolver=Resolver(; depth=depth), kwargs...)
end

end
