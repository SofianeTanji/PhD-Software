@enum ObjectiveOperation::UInt8 begin
    LeafOperation
    SumOperation
    DifferenceOperation
    ScaleOperation
    CompositionOperation
    MaximumOperation
    MinimumOperation
end

"""The strength of an oracle's accuracy contract."""
@enum OracleExactness::UInt8 begin
    ExactOracle = 0
    ControlledInexactOracle = 1
    InexactOracle = 2
    AnyOracleExactness = 3
end

const _ORACLE_EXACTNESS_BY_NAME = Dict(
    :exact => ExactOracle,
    :controlled_inexact => ControlledInexactOracle,
    :inexact => InexactOracle,
    :any => AnyOracleExactness,
)
const _ORACLE_EXACTNESS_NAMES = Dict(
    value => name for (name, value) in _ORACLE_EXACTNESS_BY_NAME
)

function oracle_exactness(name::Symbol)
    return get(_ORACLE_EXACTNESS_BY_NAME, name) do
        throw(ArgumentError("unsupported oracle exactness: $name"))
    end
end

oracle_exactness(value::OracleExactness) = value
exactness_name(value::OracleExactness) = _ORACLE_EXACTNESS_NAMES[value]

function _exactness_satisfies(available::OracleExactness, required::OracleExactness)
    available === AnyOracleExactness && return false
    return UInt8(available) <= UInt8(required)
end

function _combined_oracle_exactness(values)
    isempty(values) && throw(ArgumentError("cannot combine an empty oracle collection"))
    result = maximum(value.exactness for value in values)
    result === AnyOracleExactness &&
        throw(ArgumentError(":any is only valid in an oracle requirement"))
    return result
end

"""A symbolic optimization variable."""
struct Variable
    name::Symbol
    shape::Vector{Int}

    function Variable(name::Symbol, shape=Int[])
        dimensions = shape isa Integer ? Int[shape] : Int[shape...]
        all(>(0), dimensions) ||
            throw(ArgumentError("variable dimensions must be positive"))
        return new(name, dimensions)
    end
end

variable(name::Symbol; shape=Int[]) = Variable(name, shape)

"""A mathematical fact attached to a term."""
struct Property
    name::Symbol
    parameters::Dict{Symbol,ScalarExpr}
end

"""An available operation and its accuracy/cost contract."""
struct Oracle
    name::Symbol
    exactness::OracleExactness
    cost::Union{Nothing,ScalarExpr}

    function Oracle(
        name::Symbol,
        exactness::OracleExactness=ExactOracle,
        cost::Union{Nothing,ScalarExpr}=nothing,
    )
        exactness === AnyOracleExactness &&
            throw(ArgumentError(":any is only valid in an oracle requirement"))
        if cost !== nothing
            numeric = try_evaluate_scalar(cost)
            numeric === nothing ||
                (isfinite(numeric) && numeric >= 0) ||
                throw(ArgumentError("an oracle cost must be finite and nonnegative"))
        end
        return new(name, exactness, cost)
    end
end

function Oracle(name::Symbol, exactness::Symbol, cost::Union{Nothing,ScalarExpr}=nothing)
    return Oracle(name, oracle_exactness(exactness), cost)
end

const _PROPERTY_PARAMETER = Dict(
    :smooth => :L,
    :strongly_convex => :mu,
    :polyak_lojasiewicz => :mu,
    :gradient_mapping_dominance => :mu,
    :centered_quadratic => :coefficient,
    :hypo_convex => :rho,
    :lipschitz => :M,
    :compact_domain => :diameter,
    :frank_wolfe_curvature => :curvature,
)

const _NONNEGATIVE_PROPERTY_PARAMETERS = Set((
    (:smooth, :L),
    (:hypo_convex, :rho),
    (:lipschitz, :M),
    (:compact_domain, :diameter),
    (:frank_wolfe_curvature, :curvature),
    (:linear, :operator_norm),
    (:linear, :min_singular_value),
    (:linear, :frame_constant),
    (:linear_composition, :operator_norm),
))

const _POSITIVE_PROPERTY_PARAMETERS = Set((
    (:strongly_convex, :mu),
    (:polyak_lojasiewicz, :mu),
    (:gradient_mapping_dominance, :mu),
    (:centered_quadratic, :coefficient),
))

function _validate_property_parameter(property_name::Symbol, parameter::Symbol, value)
    numeric = if value isa Real
        value
    elseif value isa ScalarExpr
        try_evaluate_scalar(value)
    else
        nothing
    end
    numeric === nothing && return value
    key = (property_name, parameter)
    if key in _NONNEGATIVE_PROPERTY_PARAMETERS
        isfinite(numeric) && numeric >= 0 ||
            throw(ArgumentError("$property_name.$parameter must be finite and nonnegative"))
    elseif key in _POSITIVE_PROPERTY_PARAMETERS
        isfinite(numeric) && numeric > 0 ||
            throw(ArgumentError("$property_name.$parameter must be finite and positive"))
    end
    return value
end

function _parameter_scalar(name::Symbol, value)
    value === nothing && return _automatic_scalar(name)
    return scalar(value)
end

function property(name::Symbol; kwargs...)
    parameters = Dict{Symbol,ScalarExpr}()
    for (parameter, value) in pairs(kwargs)
        _validate_property_parameter(name, parameter, value)
        parameters[parameter] = _parameter_scalar(parameter, value)
    end
    return Property(name, parameters)
end

function property(name::Symbol, value; parameter::Union{Nothing,Symbol}=nothing)
    parameter_name = something(parameter, get(_PROPERTY_PARAMETER, name, :value))
    _validate_property_parameter(name, parameter_name, value)
    return Property(name, Dict(parameter_name => _parameter_scalar(parameter_name, value)))
end

function oracle(
    name::Symbol;
    exactness::Union{Symbol,OracleExactness}=ExactOracle,
    cost::Union{Nothing,Real,ScalarExpr}=nothing,
)
    cost isa Real &&
        (!isfinite(cost) || cost < 0) &&
        throw(ArgumentError("an oracle cost must be finite and nonnegative"))
    return Oracle(name, exactness, cost === nothing ? nothing : scalar(cost))
end

convex() = property(:convex)
smooth(L=nothing) = property(:smooth; L=L)
strongly_convex(mu=nothing) = property(:strongly_convex; mu=mu)
polyak_lojasiewicz(mu=nothing) = property(:polyak_lojasiewicz; mu=mu)
gradient_mapping_dominance(mu=nothing) = property(:gradient_mapping_dominance; mu=mu)
hypo_convex(rho=nothing) = property(:hypo_convex; rho=rho)
lipschitz(M=nothing) = property(:lipschitz; M=M)
compact_domain(diameter=nothing) = property(:compact_domain; diameter=diameter)
function frank_wolfe_curvature(curvature=nothing)
    property(:frank_wolfe_curvature; curvature=curvature)
end
function quadratic(; lambda_min=nothing, lambda_max=nothing)
    numeric_min = lambda_min isa ScalarExpr ? try_evaluate_scalar(lambda_min) : lambda_min
    numeric_max = lambda_max isa ScalarExpr ? try_evaluate_scalar(lambda_max) : lambda_max
    for (name, value) in ((:lambda_min, numeric_min), (:lambda_max, numeric_max))
        value isa Real &&
            !isfinite(value) &&
            throw(ArgumentError("quadratic.$name must be finite"))
    end
    numeric_min isa Real &&
        numeric_max isa Real &&
        numeric_min > numeric_max &&
        throw(ArgumentError("quadratic.lambda_min must not exceed lambda_max"))
    parameters = Dict{Symbol,ScalarExpr}()
    lambda_min === nothing ||
        (parameters[:lambda_min] = _parameter_scalar(:lambda_min, lambda_min))
    lambda_max === nothing ||
        (parameters[:lambda_max] = _parameter_scalar(:lambda_max, lambda_max))
    return Property(:quadratic, parameters)
end
function centered_quadratic(coefficient=nothing)
    property(:centered_quadratic; coefficient=coefficient)
end
monotone() = property(:monotone)
function linear_composition(operator_norm=nothing)
    property(:linear_composition; operator_norm=operator_norm)
end

function linear(; operator_norm=nothing, min_singular_value=nothing, frame_constant=nothing)
    for (name, value) in (
        (:operator_norm, operator_norm),
        (:min_singular_value, min_singular_value),
        (:frame_constant, frame_constant),
    )
        _validate_property_parameter(:linear, name, value)
    end
    parameters = Dict(:operator_norm => _parameter_scalar(:operator_norm, operator_norm))
    min_singular_value === nothing || (
        parameters[:min_singular_value] = _parameter_scalar(
            :min_singular_value, min_singular_value
        )
    )
    frame_constant === nothing ||
        (parameters[:frame_constant] = _parameter_scalar(:frame_constant, frame_constant))
    return Property(:linear, parameters)
end

const _TERM_SEQUENCE = Ref{UInt64}(0)

function _term_scope(name::Symbol)
    _TERM_SEQUENCE[] += 1
    return string(name, '#', _TERM_SEQUENCE[])
end

function _scope_property(value::Property, scope::String)
    parameters = Dict(
        name => scope_scalars(parameter, scope) for (name, parameter) in value.parameters
    )
    return Property(value.name, parameters)
end

"""A uniform node in the closed objective-expression algebra."""
struct Term
    name::Symbol
    scope::String
    operation::ObjectiveOperation
    children::Vector{Term}
    coefficient::Union{Nothing,ScalarExpr}
    variables::Vector{Variable}
    properties::Vector{Property}
    oracles::Vector{Oracle}
end

function _fact_vector(value, ::Type{T}) where {T}
    value === nothing && return T[]
    value isa T && return T[value]
    return T[item for item in value]
end

function term(name::Symbol, variables::Variable...; properties=Property[], oracles=Oracle[])
    scope = _term_scope(name)
    direct_properties = Property[
        _scope_property(item, scope) for item in _fact_vector(properties, Property)
    ]
    direct_oracles = _fact_vector(oracles, Oracle)
    return Term(
        name,
        scope,
        LeafOperation,
        Term[],
        nothing,
        Variable[variables...],
        direct_properties,
        direct_oracles,
    )
end

function _operation_term(
    name::Symbol,
    operation::ObjectiveOperation,
    children::Vector{Term};
    coefficient::Union{Nothing,ScalarExpr}=nothing,
)
    scope = _term_scope(name)
    return Term(
        name, scope, operation, children, coefficient, Variable[], Property[], Oracle[]
    )
end

function _replace_named(values::Vector{T}, addition::T) where {T}
    name = getfield(addition, :name)
    result = T[value for value in values if getfield(value, :name) !== name]
    push!(result, addition)
    return result
end

function with(value::Term, facts::Union{Property,Oracle}...)
    properties = copy(value.properties)
    oracles = copy(value.oracles)
    for fact in facts
        if fact isa Property
            properties = _replace_named(properties, _scope_property(fact, value.scope))
        else
            oracles = _replace_named(oracles, fact)
        end
    end
    return Term(
        value.name,
        value.scope,
        value.operation,
        copy(value.children),
        value.coefficient,
        copy(value.variables),
        properties,
        oracles,
    )
end

direct_properties(value::Term) = copy(value.properties)
direct_oracles(value::Term) = copy(value.oracles)

function Base.:+(left::Term, right::Term)
    children = Term[]
    if left.operation === SumOperation
        append!(children, left.children)
    else
        push!(children, left)
    end
    if right.operation === SumOperation
        append!(children, right.children)
    else
        push!(children, right)
    end
    return _operation_term(:sum, SumOperation, children)
end

function Base.:-(left::Term, right::Term)
    _operation_term(:difference, DifferenceOperation, Term[left, right])
end

function Base.:*(coefficient::Union{Real,ScalarExpr}, value::Term)
    return _operation_term(
        :scale, ScaleOperation, Term[value]; coefficient=scalar(coefficient)
    )
end

Base.:*(value::Term, coefficient::Union{Real,ScalarExpr}) = coefficient * value

function compose(outer::Term, inner::Term)
    _operation_term(:composition, CompositionOperation, Term[outer, inner])
end

function maximum_term(first::Term, rest::Term...)
    children = Term[first, rest...]
    return _operation_term(:maximum, MaximumOperation, children)
end

function minimum_term(first::Term, rest::Term...)
    children = Term[first, rest...]
    return _operation_term(:minimum, MinimumOperation, children)
end

Base.maximum(first::Term, rest::Term...) = maximum_term(first, rest...)
Base.minimum(first::Term, rest::Term...) = minimum_term(first, rest...)

function (value::Term)(variables::Variable...)
    value.operation === LeafOperation ||
        throw(ArgumentError("only a leaf term can be bound directly to variables"))
    return Term(
        value.name,
        value.scope,
        value.operation,
        Term[],
        value.coefficient,
        Variable[variables...],
        copy(value.properties),
        copy(value.oracles),
    )
end

(outer::Term)(inner::Term) = compose(outer, inner)

struct LinearApplication
    operator::Term
end

function Base.:*(operator::Term, variable::Variable)
    has_property(operator, :linear) ||
        throw(ArgumentError("only a declared linear term can be applied to a variable"))
    return LinearApplication(operator(variable))
end

(outer::Term)(application::LinearApplication) = compose(outer, application.operator)

"""A first-order minimization problem."""
struct Problem
    objective::Term
end

minimize(objective::Term) = Problem(objective)

function objective_terms(value::Problem)
    value.objective.operation === SumOperation && return copy(value.objective.children)
    return Term[value.objective]
end

function Base.:(==)(left::Variable, right::Variable)
    return left.name === right.name && left.shape == right.shape
end

function Base.:(==)(left::Property, right::Property)
    return left.name === right.name && left.parameters == right.parameters
end

function Base.:(==)(left::Oracle, right::Oracle)
    return left.name === right.name &&
           left.exactness === right.exactness &&
           left.cost == right.cost
end

function Base.:(==)(left::Term, right::Term)
    return left.name === right.name &&
           left.scope == right.scope &&
           left.operation === right.operation &&
           left.children == right.children &&
           left.coefficient == right.coefficient &&
           left.variables == right.variables &&
           left.properties == right.properties &&
           left.oracles == right.oracles
end

Base.:(==)(left::Problem, right::Problem) = left.objective == right.objective

function Base.show(io::IO, value::Variable)
    print(io, value.name)
    isempty(value.shape) && return nothing
    print(io, '[')
    for (index, dimension) in enumerate(value.shape)
        index > 1 && print(io, ", ")
        print(io, "1:", dimension)
    end
    return print(io, ']')
end

function _show_property(io::IO, value::Property)
    print(io, value.name)
    isempty(value.parameters) && return nothing
    print(io, '(')
    for (index, (name, parameter)) in enumerate(sort!(collect(value.parameters); by=first))
        index > 1 && print(io, ", ")
        print(io, name, '=')
        show(io, parameter)
    end
    return print(io, ')')
end

Base.show(io::IO, value::Property) = _show_property(io, value)

function Base.show(io::IO, value::Oracle)
    print(io, value.name)
    value.exactness === ExactOracle || print(io, '[', exactness_name(value.exactness), ']')
    return nothing
end

const _TERM_PRECEDENCE = Dict(
    SumOperation         => 1,
    DifferenceOperation  => 1,
    ScaleOperation       => 2,
    CompositionOperation => 3,
    MaximumOperation     => 4,
    MinimumOperation     => 4,
    LeafOperation        => 5,
)

function _show_term(io::IO, value::Term, parent_precedence::Int=0)
    if value.operation === LeafOperation
        print(io, value.name)
        if !isempty(value.variables)
            print(io, '(')
            for (index, variable) in enumerate(value.variables)
                index > 1 && print(io, ", ")
                show(io, variable)
            end
            print(io, ')')
        end
        return nothing
    end

    precedence = _TERM_PRECEDENCE[value.operation]
    wrap = precedence < parent_precedence
    wrap && print(io, '(')
    if value.operation === ScaleOperation
        show(io, value.coefficient)
        print(io, " * ")
        _show_term(io, only(value.children), precedence)
    elseif value.operation === CompositionOperation
        _show_term(io, value.children[1], precedence)
        print(io, " ∘ ")
        _show_term(io, value.children[2], precedence)
    elseif value.operation in (MaximumOperation, MinimumOperation)
        print(io, value.operation === MaximumOperation ? "maximum(" : "minimum(")
        for (index, child) in enumerate(value.children)
            index > 1 && print(io, ", ")
            _show_term(io, child)
        end
        print(io, ')')
    else
        separator = value.operation === SumOperation ? " + " : " - "
        for (index, child) in enumerate(value.children)
            index > 1 && print(io, separator)
            _show_term(io, child, precedence + 1)
        end
    end
    wrap && print(io, ')')
    return nothing
end

Base.show(io::IO, value::Term) = _show_term(io, value)

function Base.show(io::IO, value::Problem)
    print(io, "minimize(")
    show(io, value.objective)
    return print(io, ')')
end
