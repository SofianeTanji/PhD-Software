"""A symbol whose identity is tied to a mathematical declaration."""
struct ScalarSymbol
    scope::String
    name::Symbol
end

function Base.:(==)(left::ScalarSymbol, right::ScalarSymbol)
    left.scope == right.scope && left.name === right.name
end
function Base.isequal(left::ScalarSymbol, right::ScalarSymbol)
    isequal(left.scope, right.scope) && left.name === right.name
end
Base.hash(value::ScalarSymbol, seed::UInt) = hash(value.name, hash(value.scope, seed))

const _SCALAR_HEADS = Set((
    :literal,
    :symbol,
    :auto,
    :add,
    :subtract,
    :multiply,
    :divide,
    :power,
    :negate,
    :sqrt,
    :log,
    :exp,
    :abs,
    :min,
    :max,
))

"""
    ScalarExpr

A deliberately small scalar expression used for constants, method parameters,
bounds, and oracle complexities. Every expression has the same Julia type.
"""
struct ScalarExpr <: Number
    head::Symbol
    payload::Any
    args::Vector{ScalarExpr}

    function ScalarExpr(head::Symbol, payload=nothing, args=ScalarExpr[])
        head in _SCALAR_HEADS || throw(ArgumentError("unknown scalar operation: $head"))
        head === :literal &&
            !(payload isa Real) &&
            throw(ArgumentError("a scalar literal must contain a real number"))
        head === :symbol &&
            !(payload isa ScalarSymbol) &&
            throw(ArgumentError("a symbolic scalar must contain a ScalarSymbol"))
        head === :auto &&
            !(payload isa Symbol) &&
            throw(ArgumentError("an automatic scalar must contain a parameter name"))
        return new(head, payload, collect(args))
    end
end

scalar(value::ScalarExpr) = value
scalar(value::Real) = ScalarExpr(:literal, value)

"""Create a scoped symbolic scalar."""
function symbolic(name::Symbol; scope::Union{Symbol,AbstractString}=:global)
    ScalarExpr(:symbol, ScalarSymbol(string(scope), name))
end

_automatic_scalar(name::Symbol) = ScalarExpr(:auto, name)

Base.convert(::Type{ScalarExpr}, value::Real) = scalar(value)
Base.promote_rule(::Type{ScalarExpr}, ::Type{<:Real}) = ScalarExpr
Base.zero(::Type{ScalarExpr}) = scalar(0)
Base.one(::Type{ScalarExpr}) = scalar(1)
Base.zero(::ScalarExpr) = scalar(0)
Base.one(::ScalarExpr) = scalar(1)

_is_literal(value::ScalarExpr) = value.head === :literal
_literal(value::ScalarExpr) = value.payload
Base.iszero(value::ScalarExpr) = _is_literal(value) && iszero(_literal(value))
Base.isone(value::ScalarExpr) = _is_literal(value) && isone(_literal(value))

function _literal_binary(head::Symbol, left::Real, right::Real)
    head === :add && return left + right
    head === :subtract && return left - right
    head === :multiply && return left * right
    head === :divide && return left / right
    head === :power && return left^right
    head === :min && return min(left, right)
    head === :max && return max(left, right)
    throw(ArgumentError("unsupported scalar binary operation: $head"))
end

function _binary_scalar(head::Symbol, left, right)
    a = scalar(left)
    b = scalar(right)
    if _is_literal(a) && _is_literal(b)
        return scalar(_literal_binary(head, _literal(a), _literal(b)))
    end
    head === :add && iszero(a) && return b
    head === :add && iszero(b) && return a
    head === :subtract && iszero(b) && return a
    head === :multiply && (iszero(a) || iszero(b)) && return scalar(0)
    head === :multiply && isone(a) && return b
    head === :multiply && isone(b) && return a
    head === :divide && isone(b) && return a
    head === :power && iszero(b) && return scalar(1)
    head === :power && isone(b) && return a
    return ScalarExpr(head, nothing, ScalarExpr[a, b])
end

Base.:+(left::ScalarExpr, right::ScalarExpr) = _binary_scalar(:add, left, right)
Base.:+(left::ScalarExpr, right::Real) = _binary_scalar(:add, left, right)
Base.:+(left::Real, right::ScalarExpr) = _binary_scalar(:add, left, right)
Base.:-(left::ScalarExpr, right::ScalarExpr) = _binary_scalar(:subtract, left, right)
Base.:-(left::ScalarExpr, right::Real) = _binary_scalar(:subtract, left, right)
Base.:-(left::Real, right::ScalarExpr) = _binary_scalar(:subtract, left, right)
Base.:*(left::ScalarExpr, right::ScalarExpr) = _binary_scalar(:multiply, left, right)
Base.:*(left::ScalarExpr, right::Real) = _binary_scalar(:multiply, left, right)
Base.:*(left::Real, right::ScalarExpr) = _binary_scalar(:multiply, left, right)
Base.:/(left::ScalarExpr, right::ScalarExpr) = _binary_scalar(:divide, left, right)
Base.:/(left::ScalarExpr, right::Real) = _binary_scalar(:divide, left, right)
Base.:/(left::Real, right::ScalarExpr) = _binary_scalar(:divide, left, right)
Base.:^(left::ScalarExpr, right::ScalarExpr) = _binary_scalar(:power, left, right)
Base.:^(left::ScalarExpr, right::Integer) = _binary_scalar(:power, left, right)
Base.:^(left::ScalarExpr, right::Rational) = _binary_scalar(:power, left, right)
Base.:^(left::ScalarExpr, right::Real) = _binary_scalar(:power, left, right)
Base.:^(left::Real, right::ScalarExpr) = _binary_scalar(:power, left, right)
Base.:^(left::AbstractIrrational, right::ScalarExpr) = _binary_scalar(:power, left, right)
Base.:^(left::Irrational{:ℯ}, right::ScalarExpr) = _binary_scalar(:power, left, right)

function Base.:-(value::ScalarExpr)
    _is_literal(value) && return scalar(-_literal(value))
    value.head === :negate && return only(value.args)
    return ScalarExpr(:negate, nothing, ScalarExpr[value])
end

function _unary_scalar(head::Symbol, value::ScalarExpr)
    if _is_literal(value)
        literal = _literal(value)
        head === :sqrt && return scalar(sqrt(literal))
        head === :log && return scalar(log(literal))
        head === :exp && return scalar(exp(literal))
        head === :abs && return scalar(abs(literal))
    end
    return ScalarExpr(head, nothing, ScalarExpr[value])
end

Base.sqrt(value::ScalarExpr) = _unary_scalar(:sqrt, value)
Base.log(value::ScalarExpr) = _unary_scalar(:log, value)
Base.exp(value::ScalarExpr) = _unary_scalar(:exp, value)
Base.abs(value::ScalarExpr) = _unary_scalar(:abs, value)
Base.min(left::ScalarExpr, right::ScalarExpr) = _binary_scalar(:min, left, right)
Base.min(left::ScalarExpr, right::Real) = _binary_scalar(:min, left, right)
Base.min(left::Real, right::ScalarExpr) = _binary_scalar(:min, left, right)
Base.max(left::ScalarExpr, right::ScalarExpr) = _binary_scalar(:max, left, right)
Base.max(left::ScalarExpr, right::Real) = _binary_scalar(:max, left, right)
Base.max(left::Real, right::ScalarExpr) = _binary_scalar(:max, left, right)

function Base.:(==)(left::ScalarExpr, right::ScalarExpr)
    left.head === right.head && left.payload == right.payload && left.args == right.args
end
function Base.isequal(left::ScalarExpr, right::ScalarExpr)
    left.head === right.head &&
        isequal(left.payload, right.payload) &&
        isequal(left.args, right.args)
end
function Base.hash(value::ScalarExpr, seed::UInt)
    hash(value.args, hash(value.payload, hash(value.head, seed)))
end

"""Return all scoped symbols occurring in an expression."""
function scalar_symbols(expression::ScalarExpr)
    result = Set{ScalarSymbol}()
    _collect_scalar_symbols!(result, expression)
    return result
end

function _collect_scalar_symbols!(result::Set{ScalarSymbol}, expression::ScalarExpr)
    expression.head === :symbol && push!(result, expression.payload)
    for argument in expression.args
        _collect_scalar_symbols!(result, argument)
    end
    return result
end

"""Replace automatic parameter placeholders with symbols scoped to `scope`."""
function scope_scalars(expression::ScalarExpr, scope::Union{Symbol,AbstractString})
    expression.head === :auto && return symbolic(expression.payload; scope=scope)
    isempty(expression.args) && return expression
    return ScalarExpr(
        expression.head,
        expression.payload,
        ScalarExpr[scope_scalars(argument, scope) for argument in expression.args],
    )
end

function _scalar_replacements(expressions, values::AbstractDict)
    by_name = Dict{Symbol,Set{ScalarSymbol}}()
    for expression in expressions
        for symbol in scalar_symbols(expression)
            push!(get!(by_name, symbol.name, Set{ScalarSymbol}()), symbol)
        end
    end
    replacements = Dict{ScalarSymbol,Any}()
    for (name, value) in pairs(values)
        name isa Symbol || continue
        candidates = get(by_name, name, Set{ScalarSymbol}())
        length(candidates) == 1 || continue
        replacements[only(candidates)] = value
    end
    for (symbol, value) in pairs(values)
        symbol isa ScalarSymbol || continue
        replacements[symbol] = value
    end
    return replacements
end

function _substitute_scalars(expression::ScalarExpr, replacements::AbstractDict)
    if expression.head === :symbol
        haskey(replacements, expression.payload) || return expression
        return scalar(replacements[expression.payload])
    end
    isempty(expression.args) && return expression
    return ScalarExpr(
        expression.head,
        expression.payload,
        ScalarExpr[
            _substitute_scalars(argument, replacements) for argument in expression.args
        ],
    )
end

"""Substitute exact or unambiguous scoped symbols, preserving an expression."""
function substitute_scalars(expression::ScalarExpr, values::AbstractDict)
    replacements = _scalar_replacements((expression,), values)
    return _substitute_scalars(expression, replacements)
end

function _evaluate_binary(head::Symbol, args::Vector{<:Real})
    length(args) == 2 || throw(ArgumentError("$head expects two scalar arguments"))
    return _literal_binary(head, args[1], args[2])
end

function _evaluate_scalar(expression::ScalarExpr)
    expression.head === :literal && return expression.payload
    expression.head === :auto &&
        throw(ArgumentError("automatic scalar $(expression.payload) has not been scoped"))
    if expression.head === :symbol
        throw(KeyError(expression.payload))
    end

    arguments = Real[_evaluate_scalar(argument) for argument in expression.args]
    expression.head in (:add, :subtract, :multiply, :divide, :power, :min, :max) &&
        return _evaluate_binary(expression.head, arguments)
    length(arguments) == 1 ||
        throw(ArgumentError("$(expression.head) expects one scalar argument"))
    value = only(arguments)
    expression.head === :negate && return -value
    expression.head === :sqrt && return sqrt(value)
    expression.head === :log && return log(value)
    expression.head === :exp && return exp(value)
    expression.head === :abs && return abs(value)
    throw(ArgumentError("unsupported scalar operation: $(expression.head)"))
end

"""Evaluate a scalar expression after supplying exact or unambiguous symbols."""
function evaluate_scalar(expression::ScalarExpr, values::AbstractDict=Dict())
    return _evaluate_scalar(substitute_scalars(expression, values))
end

"""Evaluate an expression if all symbols are bound, otherwise return `nothing`."""
function try_evaluate_scalar(expression::ScalarExpr, values::AbstractDict=Dict())
    try
        return evaluate_scalar(expression, values)
    catch error
        error isa KeyError || rethrow()
        return nothing
    end
end

const _SCALAR_PRECEDENCE = Dict(
    :add      => 1,
    :subtract => 1,
    :multiply => 2,
    :divide   => 2,
    :negate   => 3,
    :power    => 4,
)

function _display_scalar_symbol(symbol::ScalarSymbol)
    symbol.scope == "global" && return string(symbol.name)
    label = first(split(symbol.scope, '#'; limit=2))
    return string(symbol.name, '_', label)
end

function _show_scalar(io::IO, expression::ScalarExpr, parent_precedence::Int=0)
    expression.head === :literal && return show(io, expression.payload)
    expression.head === :symbol &&
        return print(io, _display_scalar_symbol(expression.payload))
    expression.head === :auto && return print(io, expression.payload)

    if expression.head in (:sqrt, :log, :exp, :abs, :min, :max)
        print(io, expression.head, '(')
        for (index, argument) in enumerate(expression.args)
            index > 1 && print(io, ", ")
            _show_scalar(io, argument)
        end
        return print(io, ')')
    end

    precedence = get(_SCALAR_PRECEDENCE, expression.head, 5)
    wrap = precedence < parent_precedence
    wrap && print(io, '(')
    if expression.head === :negate
        print(io, '-')
        _show_scalar(io, only(expression.args), precedence)
    else
        operator = Dict(
            :add      => " + ",
            :subtract => " - ",
            :multiply => " * ",
            :divide   => " / ",
            :power    => "^",
        )[expression.head]
        _show_scalar(io, expression.args[1], precedence)
        print(io, operator)
        _show_scalar(
            io, expression.args[2], precedence + (expression.head === :power ? 0 : 1)
        )
    end
    wrap && print(io, ')')
    return nothing
end

Base.show(io::IO, expression::ScalarExpr) = _show_scalar(io, expression)
