"""A named map from a reformulated plan's output back to its source problem."""
struct OutputMap
    name::Symbol
    parameters::Dict{Symbol,ScalarExpr}
end

OutputMap(name::Symbol) = OutputMap(name, Dict{Symbol,ScalarExpr}())
function OutputMap(name::Symbol, parameters::AbstractDict)
    OutputMap(name, Dict{Symbol,ScalarExpr}(parameters))
end

"""
A typed conversion from a guarantee on a target formulation to one on its
source formulation.
"""
struct GuaranteeTransfer
    source_quantity::Symbol
    target_quantity::Symbol
    formula::Symbol
    parameters::Dict{Symbol,ScalarExpr}
end

function GuaranteeTransfer(
    source_quantity::Symbol,
    target_quantity::Symbol=source_quantity;
    formula::Symbol=:identity_transfer,
    parameters=Dict{Symbol,ScalarExpr}(),
)
    return GuaranteeTransfer(
        source_quantity, target_quantity, formula, Dict{Symbol,ScalarExpr}(parameters)
    )
end

"""One certified edge in a reformulation path."""
struct ReformulationStep
    rule::Symbol
    source::Problem
    target::Problem
    transfer::GuaranteeTransfer
    output_map::OutputMap
    parameters::Dict{Symbol,ScalarExpr}
    parameter_domains::Dict{Symbol,ParameterDomain}
end

function ReformulationStep(
    rule::Symbol,
    source::Problem,
    target::Problem,
    transfer::GuaranteeTransfer,
    output_map::OutputMap,
    parameters::Dict{Symbol,ScalarExpr},
)
    return ReformulationStep(
        rule,
        source,
        target,
        transfer,
        output_map,
        parameters,
        Dict{Symbol,ParameterDomain}(),
    )
end

"""A search state carrying the quantity required on its current formulation."""
struct ReformulationState
    initial::Problem
    formulation::Problem
    quantity::Symbol
    steps::Vector{ReformulationStep}
end

abstract type AbstractReformulationRule end

rule_id(rule::AbstractReformulationRule) = throw(MethodError(rule_id, (rule,)))
max_applications(::AbstractReformulationRule) = 1
reformulate(::AbstractReformulationRule, ::Problem, ::Symbol) = ReformulationStep[]

"""The convex Moreau-envelope rule with one declared positive parameter."""
struct MoreauEnvelopeRule <: AbstractReformulationRule
    lambda::ScalarExpr

    function MoreauEnvelopeRule(lambda::Union{Real,ScalarExpr}=1)
        expression = scalar(lambda)
        numeric = try_evaluate_scalar(expression)
        numeric === nothing ||
            (isfinite(numeric) && numeric > 0) ||
            throw(ArgumentError("a Moreau-envelope parameter must be finite and positive"))
        return new(expression)
    end
end

rule_id(::MoreauEnvelopeRule) = :moreau_envelope

function _moreau_envelope(rule::MoreauEnvelopeRule, problem::Problem)
    has_property(problem, :convex) || return nothing
    proximal = get_oracle(problem, :prox)
    proximal === nothing && return nothing
    proximal.exactness === ExactOracle || return nothing
    operations = Oracle[oracle(:gradient; exactness=:exact, cost=proximal.cost)]
    value = get_oracle(problem, :value)
    if value !== nothing && value.exactness === ExactOracle
        cost = if value.cost === nothing || proximal.cost === nothing
            nothing
        else
            value.cost + proximal.cost
        end
        push!(operations, oracle(:value; exactness=:exact, cost=cost))
    end
    objective = term(
        :moreau_envelope;
        properties=Property[convex(), smooth(1 / rule.lambda)],
        oracles=operations,
    )
    return minimize(objective)
end

function reformulate(rule::MoreauEnvelopeRule, problem::Problem, quantity::Symbol)
    quantity === :objective_gap || return ReformulationStep[]
    target = _moreau_envelope(rule, problem)
    target === nothing && return ReformulationStep[]
    parameters = Dict(:lambda => rule.lambda)
    transfer = GuaranteeTransfer(
        :objective_gap, :objective_gap; formula=:identity_transfer, parameters=parameters
    )
    output = OutputMap(:prox, parameters)
    return ReformulationStep[ReformulationStep(
        rule_id(rule), problem, target, transfer, output, parameters
    ),]
end

default_reformulations() = AbstractReformulationRule[MoreauEnvelopeRule()]

function _scalar_structure(value::ScalarExpr)
    payload = if value.head === :symbol
        symbol = value.payload
        (first(split(symbol.scope, '#'; limit=2)), symbol.name)
    else
        value.payload
    end
    return (
        value.head, payload, Tuple(_scalar_structure(argument) for argument in value.args)
    )
end

function _property_structure(value::Property)
    parameters = Tuple(
        (name, _scalar_structure(parameter)) for
        (name, parameter) in sort!(collect(value.parameters); by=first)
    )
    return (value.name, parameters)
end

function _term_structure(value::Term)
    property_facts = sort!(_property_structure.(value.properties); by=first)
    oracle_facts = sort!(
        [
            (
                fact.name,
                fact.exactness,
                fact.cost === nothing ? nothing : _scalar_structure(fact.cost),
            ) for fact in value.oracles
        ];
        by=first,
    )
    return (
        value.name,
        value.operation,
        value.coefficient === nothing ? nothing : _scalar_structure(value.coefficient),
        Tuple(_term_structure(child) for child in value.children),
        Tuple((variable.name, Tuple(variable.shape)) for variable in value.variables),
        Tuple(property_facts),
        Tuple(oracle_facts),
    )
end

function _formulation_key(problem::Problem, quantity::Symbol)
    return (_term_structure(problem.objective), quantity)
end

function _rule_count(steps::Vector{ReformulationStep}, identifier::Symbol)
    return count(step -> step.rule === identifier, steps)
end

"""Enumerate the initial formulation and finite, typed reformulation paths."""
function reformulation_states(
    problem::Problem,
    quantity::Symbol;
    rules=default_reformulations(),
    formulas::FormulaRegistry,
    depth::Integer=1,
)
    depth >= 0 || throw(ArgumentError("reformulation depth must be nonnegative"))
    states = ReformulationState[ReformulationState(
        problem, problem, quantity, ReformulationStep[]
    ),]
    seen = Set{Any}((_formulation_key(problem, quantity),))
    frontier = copy(states)
    for _ in 1:depth
        next_frontier = ReformulationState[]
        for state in frontier
            for rule in rules
                rule isa AbstractReformulationRule || throw(
                    ArgumentError(
                        "reformulation rules must implement AbstractReformulationRule"
                    ),
                )
                identifier = rule_id(rule)
                _rule_count(state.steps, identifier) < max_applications(rule) || continue
                for step in reformulate(rule, state.formulation, state.quantity)
                    step.rule === identifier || throw(
                        ArgumentError(
                            "reformulation step rule does not match its provider"
                        ),
                    )
                    step.source == state.formulation || throw(
                        ArgumentError(
                            "reformulation step source does not match search state"
                        ),
                    )
                    step.transfer.source_quantity === state.quantity || throw(
                        ArgumentError(
                            "reformulation transfer has an incompatible source quantity"
                        ),
                    )
                    path = vcat(state.steps, ReformulationStep[step])
                    try
                        transfer_bound(
                            path,
                            symbolic(:transfer_probe; scope=:reformulation_validation);
                            formulas=formulas,
                        )
                    catch error
                        throw(
                            ArgumentError(
                                "reformulation rule $(identifier) produced a guarantee " *
                                "transfer that does not compose: $(sprint(showerror, error))",
                            ),
                        )
                    end
                    target_quantity = step.transfer.target_quantity
                    key = _formulation_key(step.target, target_quantity)
                    key in seen && continue
                    push!(seen, key)
                    successor = ReformulationState(
                        problem, step.target, target_quantity, path
                    )
                    push!(states, successor)
                    push!(next_frontier, successor)
                end
            end
        end
        frontier = next_frontier
        isempty(frontier) && break
    end
    return states
end

function _apply_transfer(
    transfer::GuaranteeTransfer, value::ScalarExpr, formulas::Union{Nothing,FormulaRegistry}
)
    transfer.formula === :identity_transfer && return value
    formulas === nothing && throw(
        ArgumentError(
            "guarantee-transfer formula $(transfer.formula) requires a formula registry"
        ),
    )
    implementation = formula(formulas, transfer.formula)
    arguments = Dict{Symbol,Any}(:value => value)
    merge!(arguments, transfer.parameters)
    return scalar(implementation(; arguments...))
end

function transfer_bound(
    steps::Vector{ReformulationStep},
    value::ScalarExpr;
    formulas::Union{Nothing,FormulaRegistry}=nothing,
)
    result = value
    for step in reverse(steps)
        result = _apply_transfer(step.transfer, result, formulas)
    end
    return result
end
