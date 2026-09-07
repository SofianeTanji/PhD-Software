_dsl_global(name::Symbol) = GlobalRef(@__MODULE__, name)

function _dsl_dimension(index)
    index isa Integer && return Int(index)
    return length(index)
end

_dsl_shape(indices::Tuple) = Int[_dsl_dimension(index) for index in indices]

function _dsl_declaration(spec, macro_name::AbstractString)
    if spec isa Symbol
        return spec, ()
    elseif spec isa Expr && spec.head === :ref
        name = first(spec.args)
        name isa Symbol ||
            throw(ArgumentError("$macro_name expects a symbolic declaration"))
        return name, Tuple(spec.args[2:end])
    elseif spec isa Expr && spec.head === :call
        name = first(spec.args)
        name isa Symbol || throw(ArgumentError("$macro_name expects a named term"))
        return name, ()
    end
    throw(ArgumentError("unsupported $macro_name declaration"))
end

function _dsl_shape_expression(indices::Tuple)
    isempty(indices) && return :(Int[])
    escaped = map(esc, indices)
    return :($(_dsl_global(:_dsl_shape))(($(escaped...),)))
end

function _dsl_property_expression(annotation)
    annotation isa Symbol && return :($(_dsl_global(annotation))())
    annotation isa Expr && annotation.head === :call && return esc(annotation)
    throw(ArgumentError("an assumption must be a property name or constructor"))
end

function _dsl_oracle_expression(annotation)
    annotation isa Symbol && return :($(_dsl_global(:oracle))($(QuoteNode(annotation))))
    if annotation isa Expr && annotation.head === :call
        name = first(annotation.args)
        name isa Symbol || throw(ArgumentError("an oracle must have a symbolic name"))
        arguments = map(esc, annotation.args[2:end])
        return Expr(:call, _dsl_global(:oracle), QuoteNode(name), arguments...)
    end
    throw(ArgumentError("an oracle declaration must be a name or constructor"))
end

function _dsl_annotation_expression(subject, annotations, builder)
    subject isa Symbol ||
        throw(ArgumentError("DSL annotations require the declared term name"))
    facts = map(builder, annotations)
    return :($(esc(subject)) = $(_dsl_global(:with))($(esc(subject)), $(facts...)))
end

"""Group a familiar modeling block whose last expression is a `Problem`."""
macro model(block)
    return esc(block)
end

"""Declare a scalar, vector, or array-shaped symbolic variable."""
macro variable(spec)
    name, indices = _dsl_declaration(spec, "@variable")
    shape = _dsl_shape_expression(indices)
    return :($(esc(name)) = $(_dsl_global(:variable))($(QuoteNode(name)); shape=($shape)))
end

"""Declare a symbolic objective term; arguments document intended use."""
macro term(spec)
    name, _ = _dsl_declaration(spec, "@term")
    return :($(esc(name)) = $(_dsl_global(:term))($(QuoteNode(name))))
end

"""Attach one or more mathematical properties to a declared term."""
macro assume(subject, annotations...)
    return _dsl_annotation_expression(subject, annotations, _dsl_property_expression)
end

"""Attach one or more oracle capabilities to a declared term."""
macro oracle(subject, annotations...)
    return _dsl_annotation_expression(subject, annotations, _dsl_oracle_expression)
end

"""Build a first-order minimization problem from an objective expression."""
macro minimize(expression)
    return :($(_dsl_global(:minimize))($(esc(expression))))
end

function Base.:*(matrix::AbstractMatrix, value::Variable)
    return Atoms.linear_operator(matrix) * value
end
