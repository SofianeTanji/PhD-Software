"""A concrete assignment of objective terms to a theorem's named roles."""
struct RoleBinding
    roles::Vector{Pair{Symbol,Term}}
end

function Base.getindex(binding::RoleBinding, role::Symbol)
    index = findfirst(pair -> first(pair) === role, binding.roles)
    index === nothing && throw(KeyError(role))
    last(binding.roles[index])
end

"""Evidence that a provider can satisfy one theorem-side oracle requirement."""
abstract type AbstractOracleEvidencePayload end

struct DirectEvidencePayload <: AbstractOracleEvidencePayload end

struct OracleEvidence
    requirement::OracleRequirement
    available::Oracle
    role::Symbol
    provider::Symbol
    payload::AbstractOracleEvidencePayload
end

abstract type AbstractOracleResolver end

"""The information supplied to an oracle resolver for one required operation."""
struct OracleQuery
    term::Term
    requirement::OracleRequirement
    theorem::TheoremDeclaration
    role::Symbol
    binding::RoleBinding
    constants::Dict{Symbol,ScalarExpr}
    catalogue::Catalogue
    rules::RuleSet
end

resolve_oracle(::AbstractOracleResolver, ::OracleQuery) = OracleEvidence[]

struct DirectOracleResolver <: AbstractOracleResolver end

function resolve_oracle(::DirectOracleResolver, query::OracleQuery)
    available = get_oracle(query.term, query.requirement.name; rules=query.rules)
    provider = :direct
    structured_role = any(
        role -> role.name === query.role && :linear_composition in role.properties,
        query.theorem.roles,
    )
    if available === nothing &&
        query.requirement.name === :prox &&
        structured_role &&
        query.term.operation === CompositionOperation &&
        length(query.term.children) == 2 &&
        has_property(query.term.children[2], :linear; rules=query.rules)
        available = get_oracle(query.term.children[1], :prox; rules=query.rules)
        provider = :linear_composition_outer
    end
    if available === nothing &&
        query.requirement.name in (:operator, :adjoint) &&
        structured_role &&
        query.term.operation === CompositionOperation &&
        length(query.term.children) == 2 &&
        has_property(query.term.children[2], :linear; rules=query.rules)
        source_oracle = query.requirement.name === :operator ? :value : :gradient
        source = get_oracle(query.term.children[2], source_oracle; rules=query.rules)
        if source !== nothing
            available = oracle(
                query.requirement.name; exactness=source.exactness, cost=source.cost
            )
            provider = :linear_composition_operator
        end
    end
    available === nothing && return OracleEvidence[]
    _exactness_satisfies(available.exactness, query.requirement.exactness) ||
        return OracleEvidence[]
    return OracleEvidence[OracleEvidence(
        query.requirement, available, query.role, provider, DirectEvidencePayload()
    ),]
end

"""Try several independent oracle providers and retain every distinct success."""
struct ResolverChain <: AbstractOracleResolver
    resolvers::Vector{AbstractOracleResolver}
end

"""A stable request for applicability on one or every guarantee quantity."""
struct ApplicabilityRequest
    quantity::Union{Nothing,Symbol}
    catalogue::Catalogue
    resolver::AbstractOracleResolver
    rules::RuleSet
    reformulations::Vector{AbstractReformulationRule}
    reformulation_depth::Int
    max_bindings::Int
end

function ApplicabilityRequest(;
    quantity::Union{Nothing,Symbol}=:objective_gap,
    catalogue::Catalogue=default_catalogue(),
    resolver::AbstractOracleResolver=DirectOracleResolver(),
    rules::RuleSet=DEFAULT_RULES,
    reformulations=default_reformulations(),
    reformulation_depth::Integer=1,
    max_bindings::Integer=4096,
)
    requested_quantity = quantity === :all ? nothing : quantity
    reformulation_depth >= 0 ||
        throw(ArgumentError("reformulation depth must be nonnegative"))
    max_bindings > 0 || throw(ArgumentError("max_bindings must be positive"))
    typed_reformulations = AbstractReformulationRule[]
    for rule in reformulations
        rule isa AbstractReformulationRule || throw(
            ArgumentError("reformulation rules must implement AbstractReformulationRule"),
        )
        push!(typed_reformulations, rule)
    end
    return ApplicabilityRequest(
        requested_quantity,
        catalogue,
        resolver,
        rules,
        typed_reformulations,
        Int(reformulation_depth),
        Int(max_bindings),
    )
end

function with(
    request::ApplicabilityRequest;
    quantity=request.quantity,
    catalogue=request.catalogue,
    resolver=request.resolver,
    rules=request.rules,
    reformulations=request.reformulations,
    reformulation_depth=request.reformulation_depth,
    max_bindings=request.max_bindings,
)
    return ApplicabilityRequest(;
        quantity=quantity,
        catalogue=catalogue,
        resolver=resolver,
        rules=rules,
        reformulations=reformulations,
        reformulation_depth=reformulation_depth,
        max_bindings=max_bindings,
    )
end

function ResolverChain(resolvers::AbstractOracleResolver...)
    ResolverChain(AbstractOracleResolver[resolvers...])
end

function resolve_oracle(resolver::ResolverChain, query::OracleQuery)
    result = OracleEvidence[]
    for provider in resolver.resolvers
        append!(result, resolve_oracle(provider, query))
    end
    return result
end

"""A named fixed recipe or free admissible domain selected by a theorem."""
struct ParameterPlan
    name::Symbol
    formula::Union{Nothing,Symbol}
    domain::Union{Nothing,ParameterDomain}

    function ParameterPlan(
        name::Symbol,
        formula::Union{Nothing,Symbol},
        domain::Union{Nothing,ParameterDomain},
    )
        (formula === nothing) == (domain === nothing) && throw(
            ArgumentError("a parameter plan must select exactly one recipe or domain"),
        )
        return new(name, formula, domain)
    end
end

ParameterPlan(name::Symbol, formula::Symbol) = ParameterPlan(name, formula, nothing)
ParameterPlan(name::Symbol, domain::ParameterDomain) = ParameterPlan(name, nothing, domain)

"""A resolved admissible interval for a free method parameter."""
struct ParameterInterval
    lower::ScalarExpr
    upper::ScalarExpr
    lower_closed::Bool
    upper_closed::Bool
    scale::Symbol
end

"""A non-executable method prescription backed by a theorem."""
struct MethodPlan
    method::MethodDeclaration
    parameters::Vector{ParameterPlan}
    source_formulation::Problem
    active_formulation::Problem
    binding::RoleBinding
    oracle_evidence::Vector{OracleEvidence}
end

"""The native theorem guarantee before any reformulation transfer is applied."""
struct Guarantee
    requested_quantity::Symbol
    native_quantity::Symbol
    output::Symbol
    initial_bound::Symbol
    formula::Symbol
    constants::Dict{Symbol,ScalarExpr}
    formulas::FormulaRegistry
end

"""An applicable, theorem-backed method plan for a problem."""
struct Certificate
    theorem::TheoremDeclaration
    plan::MethodPlan
    guarantee::Guarantee
    reformulation_steps::Vector{ReformulationStep}
end

method_declaration(certificate::Certificate) = certificate.plan.method
requested_quantity(certificate::Certificate) = certificate.guarantee.requested_quantity
initial_bound_quantity(certificate::Certificate) = certificate.guarantee.initial_bound
parameter_plans(certificate::Certificate) = certificate.plan.parameters
function parameter_domain(certificate::Certificate, name::Symbol)
    parameter = findfirst(value -> value.name === name, certificate.plan.parameters)
    if parameter !== nothing
        domain = certificate.plan.parameters[parameter].domain
        domain === nothing &&
            throw(ArgumentError("parameter $name has a fixed theorem recipe"))
        return domain
    end
    domains = ParameterDomain[
        step.parameter_domains[name] for step in certificate.reformulation_steps if
        haskey(step.parameter_domains, name)
    ]
    isempty(domains) && throw(KeyError(name))
    all(domain -> domain == first(domains), domains) ||
        throw(ArgumentError("parameter $name has incompatible reformulation domains"))
    return first(domains)
end
source_formulation(certificate::Certificate) = certificate.plan.source_formulation
active_formulation(certificate::Certificate) = certificate.plan.active_formulation
role_binding(certificate::Certificate) = certificate.plan.binding
oracle_evidence(certificate::Certificate) = certificate.plan.oracle_evidence
reformulation_path(certificate::Certificate) = certificate.reformulation_steps
function has_reformulation(certificate::Certificate, rule::Symbol)
    any(step -> step.rule === rule, certificate.reformulation_steps)
end
function same_formulation(left::Certificate, right::Certificate)
    return left.plan.source_formulation == right.plan.source_formulation &&
           _term_structure(left.plan.active_formulation.objective) ==
           _term_structure(right.plan.active_formulation.objective)
end

function _uses_scalar_name(expression::ScalarExpr, name::Symbol)
    return any(symbol -> symbol.name === name, scalar_symbols(expression))
end

"""Return whether a certificate depends on a scoped scalar with `name`."""
function uses_scalar(certificate::Certificate, name::Symbol)
    selected_parameters = Set(parameter.name for parameter in certificate.plan.parameters)
    any(
        parameter -> parameter.name === name && parameter.domain !== nothing,
        certificate.plan.parameters,
    ) && return true
    any(
        expression -> _uses_scalar_name(expression, name),
        values(certificate.guarantee.constants),
    ) && return true
    for evidence in certificate.plan.oracle_evidence
        cost = evidence.available.cost
        if cost !== nothing && _uses_scalar_name(cost, name)
            return true
        end
        contract = evidence.requirement.inexactness
        if contract !== nothing
            _uses_scalar_name(contract.inner.initial_value, name) && return true
            evidence.available.exactness !== ExactOracle &&
                contract.accuracy === name &&
                name ∉ selected_parameters &&
                return true
            !haskey(certificate.guarantee.constants, contract.kappa) &&
                contract.kappa === name &&
                name ∉ selected_parameters &&
                return true
        end
    end
    for step in certificate.reformulation_steps
        for parameters in
            (step.parameters, step.transfer.parameters, step.output_map.parameters)
            any(expression -> _uses_scalar_name(expression, name), values(parameters)) &&
                return true
        end
    end
    return false
end

function _flatten_objective(value::Term)
    value.operation === SumOperation || return Term[value]
    result = Term[]
    for child in value.children
        append!(result, _flatten_objective(child))
    end
    return result
end

function _group_term(values::Vector{Term})
    length(values) == 1 && return only(values)
    return reduce(+, values)
end

function _binding_structure(binding::RoleBinding)
    return Tuple(
        role => Tuple(term.scope for term in _flatten_objective(value)) for
        (role, value) in binding.roles
    )
end

function _role_bindings(problem::Problem, theorem::TheoremDeclaration, max_bindings::Int)
    role_count = length(theorem.roles)
    role_count > 0 || return RoleBinding[]
    role_count == 1 && return RoleBinding[RoleBinding(
        Pair{Symbol,Term}[only(theorem.roles).name => problem.objective]
    ),]
    terms = _flatten_objective(problem.objective)
    length(terms) >= role_count || return RoleBinding[]
    groups = [Term[] for _ in 1:role_count]
    result = RoleBinding[]
    seen = Set{Any}()
    visited = Ref(0)

    function search(index::Int)
        if index > length(terms)
            all(!isempty, groups) || return nothing
            binding = RoleBinding(
                Pair{Symbol,Term}[
                    theorem.roles[role].name => _group_term(groups[role]) for
                    role in 1:role_count
                ],
            )
            key = _binding_structure(binding)
            key in seen && return nothing
            push!(seen, key)
            push!(result, binding)
            return nothing
        end

        remaining_after = length(terms) - index
        for role in 1:role_count
            visited[] += 1
            visited[] <= max_bindings || throw(
                ArgumentError(
                    "objective decomposition exceeded $max_bindings backtracking " *
                    "nodes for theorem $(theorem.id); increase max_bindings " *
                    "explicitly if this is intentional",
                ),
            )
            push!(groups[role], terms[index])
            empty_roles = count(isempty, groups)
            empty_roles <= remaining_after && search(index + 1)
            pop!(groups[role])
        end
        return nothing
    end

    search(1)
    return result
end

function _known_property_parameter(property_name::Symbol)
    return get(_PROPERTY_PARAMETER, property_name, nothing)
end

function _put_constant!(constants, name, value)
    if haskey(constants, name)
        constants[name] == value || return false
    else
        constants[name] = value
    end
    return true
end

function _match_properties(value::Term, requirement::RoleRequirement, rules::RuleSet)
    constants = Pair{FormulaInputSource,ScalarExpr}[]
    for property_name in requirement.properties
        fact = get_property(value, property_name; rules=rules)
        fact === nothing && return nothing
        if isempty(fact.parameters)
            parameter = _known_property_parameter(property_name)
            parameter === nothing || push!(
                constants,
                FormulaInputSource(requirement.name, property_name, parameter) =>
                    symbolic(parameter; scope=value.scope),
            )
        else
            for (name, parameter) in fact.parameters
                push!(
                    constants,
                    FormulaInputSource(requirement.name, property_name, name) => parameter,
                )
            end
        end
    end
    return constants
end

function _constant_context(values::Vector{Pair{FormulaInputSource,ScalarExpr}})
    constants = Dict{Symbol,ScalarExpr}()
    sources = FormulaInputSource[first(value) for value in values]
    for (alias, index) in _formula_input_aliases(sources)
        _put_constant!(constants, alias, last(values[index])) || return nothing
    end
    return constants
end

function _evidence_products(choices::Vector{Vector{OracleEvidence}})
    isempty(choices) && return [OracleEvidence[]]
    result = [OracleEvidence[]]
    for alternatives in choices
        next = Vector{OracleEvidence}[]
        for prefix in result, alternative in alternatives
            push!(next, vcat(prefix, OracleEvidence[alternative]))
        end
        result = next
    end
    return result
end

function _validate_oracle_evidence(
    evidence::OracleEvidence, requirement::OracleRequirement, role::Symbol
)
    evidence.role === role ||
        throw(ArgumentError("oracle resolver returned evidence for the wrong role"))
    evidence.requirement.name === requirement.name &&
    evidence.requirement.exactness === requirement.exactness &&
    evidence.requirement.inexactness == requirement.inexactness ||
        throw(ArgumentError("oracle resolver changed the requested requirement"))
    evidence.available.name === requirement.name ||
        throw(ArgumentError("oracle resolver returned the wrong operation"))
    _exactness_satisfies(evidence.available.exactness, requirement.exactness) ||
        throw(ArgumentError("oracle resolver returned insufficient exactness"))
    return evidence
end

function _match_binding(
    binding::RoleBinding,
    theorem::TheoremDeclaration,
    catalogue::Catalogue,
    resolver::AbstractOracleResolver,
    rules::RuleSet,
)
    property_constants = Pair{FormulaInputSource,ScalarExpr}[]
    for requirement in theorem.roles
        value = binding[requirement.name]
        matched = _match_properties(value, requirement, rules)
        matched === nothing && return nothing
        append!(property_constants, matched)
    end
    constants = _constant_context(property_constants)
    constants === nothing && return nothing
    choices = Vector{OracleEvidence}[]
    for requirement in theorem.roles
        value = binding[requirement.name]
        for required_oracle in requirement.oracles
            query = OracleQuery(
                value,
                required_oracle,
                theorem,
                requirement.name,
                binding,
                copy(constants),
                catalogue,
                rules,
            )
            alternatives = resolve_oracle(resolver, query)
            isempty(alternatives) && return nothing
            for evidence in alternatives
                _validate_oracle_evidence(evidence, required_oracle, requirement.name)
            end
            push!(choices, alternatives)
        end
    end
    return constants, _evidence_products(choices)
end

function _formula_provenance_available(
    theorem::TheoremDeclaration,
    constants::Dict{Symbol,ScalarExpr},
    registry::FormulaRegistry,
)
    available = Set{Symbol}((:k, :initial))
    union!(available, keys(constants))
    union!(available, keys(theorem.parameter_rules))
    union!(available, keys(theorem.parameter_domains))
    for role in theorem.roles, requirement in role.oracles
        contract = requirement.inexactness
        contract === nothing && continue
        push!(available, contract.accuracy)
        push!(available, contract.kappa)
    end
    formula_names = Symbol[theorem.bound]
    append!(formula_names, values(theorem.parameter_rules))
    for domain in values(theorem.parameter_domains)
        domain.lower isa Symbol && push!(formula_names, domain.lower)
        domain.upper isa Symbol && push!(formula_names, domain.upper)
    end
    for formula_name in formula_names
        formula_name in registry.custom || continue
        implementation = get(registry.formulas, formula_name, nothing)
        implementation === nothing && return false
        issubset(_required_formula_inputs(implementation), available) || return false
    end
    return true
end

function _certificates_on(
    state::ReformulationState,
    requested_quantity::Symbol,
    catalogue::Catalogue,
    resolver::AbstractOracleResolver,
    rules::RuleSet,
    max_bindings::Int,
)
    result = Certificate[]
    for theorem in catalogue.theorems
        theorem.quantity === state.quantity || continue
        method = method_by_id(catalogue, theorem.method)
        method === nothing && continue
        bindings = _role_bindings(state.formulation, theorem, max_bindings)
        for binding in bindings
            matched = _match_binding(binding, theorem, catalogue, resolver, rules)
            matched === nothing && continue
            constants, evidence_sets = matched
            _formula_provenance_available(theorem, constants, catalogue.formulas) ||
                continue
            parameters = ParameterPlan[
                ParameterPlan(name, formula_name) for
                (name, formula_name) in sort!(collect(theorem.parameter_rules); by=first)
            ]
            append!(
                parameters,
                ParameterPlan[
                    ParameterPlan(name, domain) for
                    (name, domain) in sort!(collect(theorem.parameter_domains); by=first)
                ],
            )
            sort!(parameters; by=parameter -> string(parameter.name))
            guarantee = Guarantee(
                requested_quantity,
                theorem.quantity,
                theorem.output,
                theorem.initial_bound,
                theorem.bound,
                constants,
                catalogue.formulas,
            )
            for evidence in evidence_sets
                plan = MethodPlan(
                    method, parameters, state.initial, state.formulation, binding, evidence
                )
                push!(result, Certificate(theorem, plan, guarantee, copy(state.steps)))
            end
        end
    end
    return result
end

"""
    certificates(problem, request=ApplicabilityRequest())

Return only theorem-backed method plans applicable to the initial formulation
or to a finite reformulation path that transfers the requested guarantee.
"""
function certificates(problem::Problem, request::ApplicabilityRequest)
    result = Certificate[]
    quantities = if request.quantity === nothing
        sort!(
            unique(Symbol[theorem.quantity for theorem in request.catalogue.theorems]);
            by=string,
        )
    else
        Symbol[request.quantity]
    end
    for quantity in quantities
        states = reformulation_states(
            problem,
            quantity;
            rules=request.reformulations,
            formulas=request.catalogue.formulas,
            depth=request.reformulation_depth,
        )
        for state in states
            append!(
                result,
                _certificates_on(
                    state,
                    quantity,
                    request.catalogue,
                    request.resolver,
                    request.rules,
                    request.max_bindings,
                ),
            )
        end
    end
    return result
end

function certificates(problem::Problem; kwargs...)
    certificates(problem, ApplicabilityRequest(; kwargs...))
end

function _initial_value(certificate::Certificate, initial_bounds)
    name = certificate.guarantee.initial_bound
    name === :none && return scalar(0)
    if haskey(initial_bounds, name)
        return scalar(initial_bounds[name])
    end
    return symbolic(name; scope=certificate.theorem.id)
end

function _substitute_context(context::AbstractDict, values)
    scalar_values = Dict{Any,Any}(
        name => value for
        (name, value) in pairs(values) if value isa Real || value isa ScalarExpr
    )
    expressions = ScalarExpr[
        value for value in Base.values(context) if value isa ScalarExpr
    ]
    replacements = _scalar_replacements(expressions, scalar_values)
    result = Dict{Symbol,Any}(
        name => value isa ScalarExpr ? _substitute_scalars(value, replacements) : value for
        (name, value) in context
    )
    for (name, replacement) in pairs(values)
        name isa Symbol && haskey(context, name) || continue
        original = context[name]
        normalized = if replacement isa Real || replacement isa ScalarExpr
            scalar(replacement)
        else
            replacement
        end
        for (alias, value) in context
            value === original && (result[alias] = normalized)
        end
    end
    return result
end

_formula_keyword(name::Symbol) = name === :gamma ? Symbol("γ") : name

function _invoke_formula(
    registry::FormulaRegistry, formula_name::Symbol, context::AbstractDict, k::Integer
)
    result, missing = _try_invoke_formula(registry, formula_name, context, k)
    missing === nothing ||
        throw(ArgumentError("formula $formula_name requires undeclared input $missing"))
    return result
end

function _try_invoke_formula(
    registry::FormulaRegistry, formula_name::Symbol, context::AbstractDict, k::Integer
)
    implementation = formula(registry, formula_name)
    arguments = Dict{Symbol,Any}()
    keywords, accepts_extra = _formula_keyword_contract(implementation)
    if accepts_extra || haskey(keywords, :k)
        arguments[get(keywords, :k, :k)] = k
    end
    for (name, value) in context
        keyword = get(keywords, name, nothing)
        if keyword !== nothing
            arguments[keyword] = value
        elseif accepts_extra
            arguments[_formula_keyword(name)] = value
        end
    end
    try
        return scalar(implementation(; arguments...)), nothing
    catch error
        if error isa UndefKeywordError
            return nothing, _context_input(error.var)
        end
        rethrow()
    end
end

function _resolve_parameter_context!(
    context::Dict{Symbol,Any},
    parameters::Vector{ParameterPlan},
    registry::FormulaRegistry,
    k::Integer,
)
    rules = Dict(
        parameter.name => parameter.formula for
        parameter in parameters if parameter.formula !== nothing
    )
    inputs = _parameter_formula_inputs(rules, registry, Dict{Symbol,Set{Symbol}}())
    order, remaining = _parameter_evaluation_order(inputs)
    isempty(remaining) || throw(
        ArgumentError(
            "circular parameter dependencies: " *
            join(sort!(string.(collect(remaining))), ", "),
        ),
    )
    for name in order
        context[name] = _invoke_formula(registry, rules[name], context, k)
    end
    for parameter in parameters
        parameter.domain === nothing && continue
        _validate_parameter_domain!(context, parameter, registry, k)
    end
    return context
end

function _parameter_endpoint(
    endpoint::Union{Float64,Symbol,ScalarExpr},
    context::Dict{Symbol,Any},
    registry::FormulaRegistry,
    k::Integer,
)
    endpoint isa Float64 && return scalar(endpoint)
    endpoint isa ScalarExpr && return endpoint
    return _invoke_formula(registry, endpoint, context, k)
end

function _resolved_parameter_interval(
    context::Dict{Symbol,Any},
    parameter::ParameterPlan,
    registry::FormulaRegistry,
    k::Integer,
)
    domain = parameter.domain
    domain === nothing && throw(ArgumentError("parameter $(parameter.name) is fixed"))
    return ParameterInterval(
        _parameter_endpoint(domain.lower, context, registry, k),
        _parameter_endpoint(domain.upper, context, registry, k),
        domain.lower_closed,
        domain.upper_closed,
        domain.scale,
    )
end

function _validate_parameter_domain!(
    context::Dict{Symbol,Any},
    parameter::ParameterPlan,
    registry::FormulaRegistry,
    k::Integer,
)
    interval = _resolved_parameter_interval(context, parameter, registry, k)
    lower = try_evaluate_scalar(interval.lower)
    upper = try_evaluate_scalar(interval.upper)
    value = try_evaluate_scalar(context[parameter.name])
    if lower !== nothing && upper !== nothing
        valid_interval = lower < upper ||
                         (lower == upper && interval.lower_closed && interval.upper_closed)
        valid_interval || throw(
            DomainError(
                (lower, upper),
                "the resolved domain for parameter $(parameter.name) is empty",
            ),
        )
    end
    value === nothing && return interval
    if lower !== nothing
        admitted = interval.lower_closed ? value >= lower : value > lower
        admitted || throw(
            DomainError(
                value,
                "parameter $(parameter.name) is below its admissible lower endpoint $lower",
            ),
        )
    end
    if upper !== nothing
        admitted = interval.upper_closed ? value <= upper : value < upper
        admitted || throw(
            DomainError(
                value,
                "parameter $(parameter.name) is above its admissible upper endpoint $upper",
            ),
        )
    end
    return interval
end

function _formula_context(certificate::Certificate, initial_bounds, values)
    context = Dict{Symbol,Any}(certificate.guarantee.constants)
    context[:initial] = _initial_value(certificate, initial_bounds)
    for parameter in certificate.plan.parameters
        parameter.domain === nothing && continue
        context[parameter.name] = symbolic(parameter.name; scope=certificate.theorem.id)
    end
    for step in certificate.reformulation_steps
        for (name, value) in step.parameters
            haskey(context, name) || (context[name] = value)
        end
    end
    for evidence in certificate.plan.oracle_evidence
        evidence.requirement.exactness === ControlledInexactOracle || continue
        contract = evidence.requirement.inexactness
        contract === nothing && continue
        name = contract.accuracy
        context[name] = if evidence.available.exactness === ExactOracle
            scalar(0)
        else
            symbolic(name; scope=certificate.theorem.id)
        end
        haskey(context, contract.kappa) || (
            context[contract.kappa] = symbolic(contract.kappa; scope=certificate.theorem.id)
        )
    end
    return _substitute_context(context, values)
end

"""Resolve the admissible interval of one free theorem or reformulation parameter."""
function parameter_interval(
    certificate::Certificate,
    name::Symbol;
    k::Integer=1,
    initial_bounds=Dict(),
    values=Dict(),
)
    index = findfirst(value -> value.name === name, certificate.plan.parameters)
    parameter = if index === nothing
        ParameterPlan(name, parameter_domain(certificate, name))
    else
        certificate.plan.parameters[index]
    end
    parameter.domain === nothing &&
        throw(ArgumentError("parameter $name has a fixed theorem recipe"))
    context = _formula_context(certificate, initial_bounds, values)
    _resolve_parameter_context!(
        context, certificate.plan.parameters, certificate.guarantee.formulas, k
    )
    return _resolved_parameter_interval(
        context, parameter, certificate.guarantee.formulas, k
    )
end

function _validate_reformulation_parameter_domains!(
    certificate::Certificate, context::Dict{Symbol,Any}, registry::FormulaRegistry, k::Integer
)
    for step in certificate.reformulation_steps
        for (name, domain) in step.parameter_domains
            haskey(context, name) || (context[name] = step.parameters[name])
            _validate_parameter_domain!(context, ParameterPlan(name, domain), registry, k)
        end
    end
    return context
end

"""Evaluate one method parameter, retaining a symbolic result when needed."""
function parameter_value(
    certificate::Certificate,
    name::Symbol;
    k::Integer=1,
    initial_bounds=Dict(),
    values=Dict(),
)
    parameter = findfirst(value -> value.name === name, certificate.plan.parameters)
    parameter === nothing && parameter_domain(certificate, name)
    context = _formula_context(certificate, initial_bounds, values)
    _resolve_parameter_context!(
        context, certificate.plan.parameters, certificate.guarantee.formulas, k
    )
    _validate_reformulation_parameter_domains!(
        certificate, context, certificate.guarantee.formulas, k
    )
    return context[name]
end

"""Evaluate a certificate's transferred guarantee after `k` iterations."""
function certificate_bound(
    certificate::Certificate,
    k::Integer;
    initial_bounds=Dict(),
    values=Dict(),
    formulas::FormulaRegistry=certificate.guarantee.formulas,
)
    context = _formula_context(certificate, initial_bounds, values)
    _resolve_parameter_context!(context, certificate.plan.parameters, formulas, k)
    _validate_reformulation_parameter_domains!(certificate, context, formulas, k)
    native = _invoke_formula(formulas, certificate.guarantee.formula, context, k)
    return transfer_bound(certificate.reformulation_steps, native; formulas=formulas)
end

function Base.show(io::IO, certificate::Certificate)
    print(io, "Certificate(", certificate.plan.method.name, ", ")
    print(io, certificate.guarantee.requested_quantity, ", theorem=")
    print(io, certificate.theorem.id)
    isempty(certificate.reformulation_steps) ||
        print(io, ", reformulations=", length(certificate.reformulation_steps))
    return print(io, ')')
end
