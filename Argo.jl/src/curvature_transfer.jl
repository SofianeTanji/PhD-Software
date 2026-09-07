module CurvatureTransfer

using ..Argo

export CurvatureTransferRule, curvature_transfer

"""
One exact, parametric split of an explicit positive centered-quadratic summand.
The original unsplit formulation supplies the endpoint cases; this rule keeps
the interior `0 < rho < coefficient` as one symbolic reformulation state.
"""
struct CurvatureTransferRule <: Argo.Authoring.AbstractReformulationRule
    rho::Argo.ScalarExpr

    function CurvatureTransferRule(
        rho::Union{Real,Argo.ScalarExpr}=Argo.symbolic(
            :rho; scope=:curvature_transfer
        ),
    )
        expression = Argo.scalar(rho)
        numeric = Argo.try_evaluate_scalar(expression)
        numeric === nothing ||
            (isfinite(numeric) && numeric > 0) ||
            throw(ArgumentError("rho must be finite and positive"))
        return new(expression)
    end
end

curvature_transfer(args...) = CurvatureTransferRule(args...)

Argo.Authoring.rule_id(::CurvatureTransferRule) = :curvature_transfer

function _flatten_sum(value::Argo.Term)
    value.operation === Argo.Authoring.SumOperation || return Argo.Term[value]
    result = Argo.Term[]
    for child in value.children
        append!(result, _flatten_sum(child))
    end
    return result
end

function _variables(value::Argo.Term)
    result = Argo.Variable[]
    function collect_from(term)
        for variable in term.variables
            variable in result || push!(result, variable)
        end
        foreach(collect_from, term.children)
        return nothing
    end
    collect_from(value)
    return result
end

function _exact_quadratic_oracles(value::Argo.Term)
    result = Argo.Oracle[]
    for name in (:value, :gradient, :prox)
        available = Argo.get_oracle(value, name)
        available === nothing && return nothing
        available.exactness === Argo.Authoring.ExactOracle || return nothing
        push!(result, available)
    end
    return result
end

function _quadratic_share(
    name::Symbol,
    coefficient::Argo.ScalarExpr,
    variables::Vector{Argo.Variable},
    operations::Vector{Argo.Oracle},
)
    return Argo.term(
        name,
        variables...;
        properties=Argo.Property[
            Argo.centered_quadratic(coefficient),
            Argo.quadratic(; lambda_min=coefficient, lambda_max=coefficient),
            Argo.convex(),
            Argo.smooth(coefficient),
            Argo.strongly_convex(coefficient),
        ],
        oracles=operations,
    )
end

function Argo.Authoring.reformulate(
    rule::CurvatureTransferRule, problem::Argo.Problem, quantity::Symbol
)
    summands = _flatten_sum(problem.objective)
    length(summands) >= 2 || return Argo.Authoring.ReformulationStep[]
    candidates = Int[]
    coefficients = Dict{Int,Argo.ScalarExpr}()
    for (index, summand) in enumerate(summands)
        centered = Argo.get_property(summand, :centered_quadratic)
        centered === nothing && continue
        coefficient = get(centered.parameters, :coefficient, nothing)
        coefficient === nothing && continue
        operations = _exact_quadratic_oracles(summand)
        operations === nothing && continue
        push!(candidates, index)
        coefficients[index] = coefficient
    end
    length(candidates) == 1 || return Argo.Authoring.ReformulationStep[]

    index = only(candidates)
    source_quadratic = summands[index]
    coefficient = coefficients[index]
    known_coefficient = Argo.try_evaluate_scalar(coefficient)
    known_rho = Argo.try_evaluate_scalar(rule.rho)
    known_coefficient === nothing || known_coefficient > 0 ||
        return Argo.Authoring.ReformulationStep[]
    known_coefficient === nothing ||
        known_rho === nothing ||
        known_rho < known_coefficient ||
        return Argo.Authoring.ReformulationStep[]

    operations = something(_exact_quadratic_oracles(source_quadratic))
    variables = _variables(source_quadratic)
    transferred = _quadratic_share(
        :transferred_curvature, rule.rho, variables, operations
    )
    retained = _quadratic_share(
        :retained_curvature, coefficient - rule.rho, variables, operations
    )
    target_summands = Argo.Term[]
    for (position, summand) in enumerate(summands)
        if position == index
            push!(target_summands, transferred, retained)
        else
            push!(target_summands, summand)
        end
    end
    target = Argo.minimize(reduce(+, target_summands))
    parameters = Dict{Symbol,Argo.ScalarExpr}(:rho => rule.rho)
    domains = Dict{Symbol,Argo.Authoring.ParameterDomain}(
        :rho => Argo.Authoring.ParameterDomain(
            0.0,
            coefficient;
            lower_closed=false,
            upper_closed=false,
            scale=:linear,
        )
    )
    transfer = Argo.Authoring.GuaranteeTransfer(
        quantity, quantity; formula=:identity_transfer, parameters=parameters
    )
    return Argo.Authoring.ReformulationStep[Argo.Authoring.ReformulationStep(
        Argo.Authoring.rule_id(rule),
        problem,
        target,
        transfer,
        Argo.Authoring.OutputMap(:identity),
        parameters,
        domains,
    )]
end

end
