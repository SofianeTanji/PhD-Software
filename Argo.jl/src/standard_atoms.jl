module Atoms

using LinearAlgebra

using ..Argo:
    Oracle,
    Property,
    ScalarExpr,
    Term,
    centered_quadratic,
    compact_domain,
    convex,
    get_oracle,
    get_property,
    has_oracle,
    has_property,
    linear,
    lipschitz,
    oracle,
    property,
    quadratic,
    scalar,
    smooth,
    strongly_convex,
    term,
    try_evaluate_scalar

export affine_operator,
    berhu,
    elastic_net,
    group_lasso,
    hinge,
    huber,
    indicator,
    l1,
    l2_norm,
    least_squares,
    linear_operator,
    linf_norm,
    logistic_loss,
    log_sum_exp,
    nuclear_norm,
    poisson_loss,
    quadratic_form,
    regularize,
    ridge,
    separable_sum,
    softplus,
    sum_squares,
    tikhonov

const ScalarLike = Union{Real,ScalarExpr}

_default_full_rank_tol() = sqrt(eps(Float64))

function _positive(value::ScalarLike, label::AbstractString)
    numeric = value isa Real ? value : try_evaluate_scalar(value)
    numeric === nothing ||
        (isfinite(numeric) && numeric > 0) ||
        throw(ArgumentError("$label must be finite and positive"))
    return scalar(value)
end

function _nonnegative(value::ScalarLike, label::AbstractString)
    numeric = value isa Real ? value : try_evaluate_scalar(value)
    numeric === nothing ||
        (isfinite(numeric) && numeric >= 0) ||
        throw(ArgumentError("$label must be finite and nonnegative"))
    return scalar(value)
end

function _dimension(value::Integer)
    value > 0 || throw(ArgumentError("a dimension must be positive"))
    return Int(value)
end

function _shape(value)
    dimensions = Int[_dimension(dimension) for dimension in value]
    isempty(dimensions) && throw(ArgumentError("a shape must not be empty"))
    return dimensions
end

function _compute_singular_extremes(matrix::AbstractMatrix)
    values = svdvals(matrix)
    isempty(values) && throw(ArgumentError("cannot inspect an empty matrix"))
    return Float64(first(values)), Float64(last(values))
end

function _singular_extremes(matrix::AbstractMatrix; sigma_max=nothing, sigma_min=nothing)
    computed_max = computed_min = nothing
    if sigma_max === nothing || sigma_min === nothing
        computed_max, computed_min = _compute_singular_extremes(matrix)
    end
    largest = sigma_max === nothing ? computed_max : sigma_max
    smallest = sigma_min === nothing ? computed_min : sigma_min
    largest isa Real && largest < 0 && throw(ArgumentError("sigma_max must be nonnegative"))
    smallest isa Real &&
        smallest < 0 &&
        throw(ArgumentError("sigma_min must be nonnegative"))
    return scalar(largest), scalar(smallest)
end

function _numerically_full_column_rank(
    matrix::AbstractMatrix, sigma_max::ScalarExpr, sigma_min::ScalarExpr, tolerance::Real
)
    size(matrix, 1) >= size(matrix, 2) || return false
    sigma_max.head === :literal || return true
    sigma_min.head === :literal || return true
    return sigma_min.payload > tolerance * sigma_max.payload
end

function _gram_extremes(
    matrix::AbstractMatrix;
    sigma_max=nothing,
    sigma_min=nothing,
    full_rank_tol::Real=_default_full_rank_tol(),
)
    largest, smallest = _singular_extremes(matrix; sigma_max=sigma_max, sigma_min=sigma_min)
    full_rank = _numerically_full_column_rank(matrix, largest, smallest, full_rank_tol)
    return full_rank ? smallest^2 : scalar(0), largest^2, full_rank
end

function _tight_frame_constant(
    matrix::AbstractMatrix,
    operator_norm::ScalarExpr;
    tolerance::Real=_default_full_rank_tol(),
)
    operator_norm.head === :literal || return nothing
    rows = size(matrix, 1)
    rows <= size(matrix, 2) || return nothing
    alpha = Float64(operator_norm.payload)^2
    alpha > 0 || return nothing
    gram = matrix * matrix'
    deviation = maximum(
        abs(gram[i, j] - (i == j ? alpha : 0.0)) for i in 1:rows for j in 1:rows
    )
    return deviation <= tolerance * alpha * rows ? scalar(alpha) : nothing
end

function _linear_property(
    matrix::AbstractMatrix;
    sigma_max=nothing,
    sigma_min=nothing,
    full_rank_tol::Real=_default_full_rank_tol(),
)
    largest, smallest = _singular_extremes(matrix; sigma_max=sigma_max, sigma_min=sigma_min)
    full_rank = _numerically_full_column_rank(matrix, largest, smallest, full_rank_tol)
    frame = _tight_frame_constant(matrix, largest)
    return linear(;
        operator_norm=largest,
        min_singular_value=full_rank ? smallest : nothing,
        frame_constant=frame,
    )
end

_basic_oracles(names::Symbol...) = Oracle[oracle(name) for name in names]

function _quadratic_properties(lambda_min::ScalarExpr, lambda_max::ScalarExpr)
    facts = Property[
        quadratic(; lambda_min=lambda_min, lambda_max=lambda_max),
        smooth(max(abs(lambda_min), abs(lambda_max))),
    ]
    if lambda_min.head !== :literal || lambda_min.payload >= 0
        push!(facts, convex())
    end
    if lambda_min.head !== :literal || lambda_min.payload > 0
        push!(facts, strongly_convex(lambda_min))
    end
    return facts
end

"""Declare the data-fit atom `(1/2) * norm(A*x - b)^2`."""
function least_squares(
    matrix::AbstractMatrix,
    target::AbstractVector;
    L=nothing,
    mu=nothing,
    sigma_max=nothing,
    sigma_min=nothing,
    full_rank_tol::Real=_default_full_rank_tol(),
)
    size(matrix, 1) == length(target) ||
        throw(DimensionMismatch("matrix rows and target length must agree"))
    derived_min, derived_max, full_rank = _gram_extremes(
        matrix; sigma_max=sigma_max, sigma_min=sigma_min, full_rank_tol=full_rank_tol
    )
    lambda_max = L === nothing ? derived_max : _nonnegative(L, "L")
    lambda_min = mu === nothing ? derived_min : _nonnegative(mu, "mu")
    if mu !== nothing && mu isa Real && mu > 0 && size(matrix, 1) < size(matrix, 2)
        throw(ArgumentError("a wide least-squares matrix cannot certify strong convexity"))
    end
    mu === nothing && !full_rank && (lambda_min = scalar(0))
    return term(
        :least_squares;
        properties=_quadratic_properties(lambda_min, lambda_max),
        oracles=_basic_oracles(:value, :gradient),
    )
end

"""Declare `(1/2) * norm(x)^2`."""
function sum_squares()
    term(
        :sum_squares;
        properties=vcat(
            _quadratic_properties(scalar(1), scalar(1)),
            Property[centered_quadratic(scalar(1))],
        ),
        oracles=_basic_oracles(:value, :gradient, :prox),
    )
end

"""Declare the weighted one-norm `weight * norm(x, 1)`."""
function l1(weight::ScalarLike=1; n::Union{Nothing,Integer}=nothing)
    lambda = _positive(weight, "l1 weight")
    facts = Property[convex()]
    n === nothing || push!(facts, lipschitz(lambda * sqrt(_dimension(n))))
    return term(:l1; properties=facts, oracles=_basic_oracles(:value, :subgradient, :prox))
end

"""Declare the weighted Euclidean norm `weight * norm(x)`."""
function l2_norm(weight::ScalarLike=1)
    lambda = _positive(weight, "l2-norm weight")
    return term(
        :l2_norm;
        properties=Property[convex(), lipschitz(lambda)],
        oracles=_basic_oracles(:value, :subgradient, :prox),
    )
end

"""Declare a logistic loss and derive a global smoothness constant."""
function logistic_loss(
    matrix::AbstractMatrix,
    labels::AbstractVector;
    mean::Bool=false,
    L=nothing,
    sigma_max=nothing,
)
    size(matrix, 1) == length(labels) ||
        throw(DimensionMismatch("matrix rows and label length must agree"))
    smoothness = if L === nothing
        largest = if sigma_max === nothing
            first(_compute_singular_extremes(Diagonal(Float64.(labels)) * matrix))
        else
            scalar(sigma_max)
        end
        largest^2 / (4 * (mean ? size(matrix, 1) : 1))
    else
        _nonnegative(L, "L")
    end
    return term(
        :logistic_loss;
        properties=Property[convex(), smooth(smoothness)],
        oracles=_basic_oracles(:value, :gradient),
    )
end

function _indicator_term(name::Symbol, facts::Vector{Property}, operations::Vector{Oracle})
    return term(name; properties=facts, oracles=operations)
end

function _indicator_simplex(n::Integer)
    _dimension(n)
    return _indicator_term(
        :indicator_simplex,
        Property[convex(), compact_domain(sqrt(2.0))],
        _basic_oracles(:prox, :linear_minimization),
    )
end

function _indicator_spectrahedron(n::Integer)
    _dimension(n)
    return _indicator_term(
        :indicator_spectrahedron,
        Property[convex(), compact_domain(sqrt(2.0))],
        _basic_oracles(:prox, :linear_minimization),
    )
end

function _indicator_ball(
    kind::Symbol,
    radius::ScalarLike;
    n::Union{Nothing,Integer}=nothing,
    shape=nothing,
    lanczos_steps::Integer=15,
)
    r = _positive(radius, "indicator radius")
    name = Symbol(:indicator_, kind)
    facts = Property[convex()]
    prox_cost = nothing
    linear_minimization_cost = nothing
    if kind === :l2_ball || kind === :l1_ball || kind === :nuclear_ball
        push!(facts, compact_domain(2 * r))
    elseif kind === :linf_ball && n !== nothing
        push!(facts, compact_domain(2 * r * sqrt(_dimension(n))))
    end
    if kind === :l1_ball && n !== nothing
        dimension = _dimension(n)
        prox_cost = scalar(dimension * log2(max(dimension, 2)))
        linear_minimization_cost = scalar(dimension)
    elseif kind === :nuclear_ball && shape !== nothing
        dimensions = _shape(shape)
        length(dimensions) == 2 ||
            throw(ArgumentError("a nuclear-ball shape must have two dimensions"))
        lanczos_steps > 0 || throw(ArgumentError("lanczos_steps must be positive"))
        p = Float64(maximum(dimensions))
        q = Float64(minimum(dimensions))
        linear_minimization_cost = scalar(4 * lanczos_steps * p * q)
        prox_cost = scalar(16 * p * q^2 + 8 * q^3)
    end
    operations = Oracle[
        oracle(:prox; cost=prox_cost),
        oracle(:linear_minimization; cost=linear_minimization_cost),
    ]
    return _indicator_term(name, facts, operations)
end

function _indicator_box(lower::AbstractVector, upper::AbstractVector)
    length(lower) == length(upper) ||
        throw(DimensionMismatch("lower and upper bounds must have equal length"))
    all(lower .<= upper) ||
        throw(ArgumentError("each lower bound must not exceed its upper bound"))
    return _indicator_term(
        :indicator_box,
        Property[convex(), compact_domain(norm(upper .- lower))],
        _basic_oracles(:prox, :linear_minimization),
    )
end

function _unbounded_indicator(
    kind::Symbol, args...; A=nothing, b=nothing, a=nothing, beta=nothing
)
    if kind === :affine
        A === nothing && throw(ArgumentError("indicator(:affine) requires A"))
        b === nothing && throw(ArgumentError("indicator(:affine) requires b"))
        size(A, 1) == length(b) ||
            throw(DimensionMismatch("matrix rows and target length must agree"))
    elseif kind === :halfspace
        a === nothing && throw(ArgumentError("indicator(:halfspace) requires a"))
        beta === nothing && throw(ArgumentError("indicator(:halfspace) requires beta"))
        iszero(norm(a)) && throw(ArgumentError("a halfspace normal must be nonzero"))
    elseif kind !== :second_order_cone
        throw(ArgumentError("unknown unbounded indicator $kind"))
    end
    return _indicator_term(
        Symbol(:indicator_, kind), Property[convex()], _basic_oracles(:prox)
    )
end

"""Declare the indicator of a supported convex set."""
function indicator(
    kind::Symbol,
    args...;
    radius=nothing,
    lower=nothing,
    upper=nothing,
    n=nothing,
    shape=nothing,
    lanczos_steps::Integer=15,
    A=nothing,
    b=nothing,
    a=nothing,
    beta=nothing,
)
    kind === :simplex && return _indicator_simplex(only(args))
    kind === :spectrahedron && return _indicator_spectrahedron(only(args))
    if kind in (:l2_ball, :l1_ball, :linf_ball, :nuclear_ball)
        radius === nothing && throw(ArgumentError("indicator($kind) requires radius"))
        return _indicator_ball(kind, radius; n=n, shape=shape, lanczos_steps=lanczos_steps)
    end
    if kind === :box
        lower === nothing && throw(ArgumentError("indicator(:box) requires lower"))
        upper === nothing && throw(ArgumentError("indicator(:box) requires upper"))
        return _indicator_box(lower, upper)
    end
    return _unbounded_indicator(kind, args...; A=A, b=b, a=a, beta=beta)
end

"""Declare a Huber regression loss."""
function huber(
    matrix::AbstractMatrix,
    target::AbstractVector;
    delta::ScalarLike=1,
    L=nothing,
    sigma_max=nothing,
)
    size(matrix, 1) == length(target) ||
        throw(DimensionMismatch("matrix rows and target length must agree"))
    _positive(delta, "Huber transition")
    smoothness = if L === nothing
        largest = if sigma_max === nothing
            first(_compute_singular_extremes(matrix))
        else
            scalar(sigma_max)
        end
        largest^2
    else
        _nonnegative(L, "L")
    end
    return term(
        :huber;
        properties=Property[convex(), smooth(smoothness)],
        oracles=_basic_oracles(:value, :gradient),
    )
end

"""Declare a hinge loss."""
function hinge(matrix::AbstractMatrix, labels::AbstractVector; M=nothing, sigma_max=nothing)
    size(matrix, 1) == length(labels) ||
        throw(DimensionMismatch("matrix rows and label length must agree"))
    modulus = if M === nothing
        largest = if sigma_max === nothing
            first(_compute_singular_extremes(matrix))
        else
            scalar(sigma_max)
        end
        largest * norm(labels)
    else
        _nonnegative(M, "M")
    end
    return term(
        :hinge;
        properties=Property[convex(), lipschitz(modulus)],
        oracles=_basic_oracles(:value, :subgradient),
    )
end

"""Declare an elastic-net penalty."""
function elastic_net(l1_weight::ScalarLike=1, l2_weight::ScalarLike=1)
    first_weight = _positive(l1_weight, "elastic-net l1 weight")
    second_weight = _positive(l2_weight, "elastic-net l2 weight")
    return term(
        :elastic_net;
        properties=Property[convex(), strongly_convex(second_weight)],
        oracles=_basic_oracles(:value, :subgradient, :prox),
    )
end

"""Declare a weighted nuclear norm."""
function nuclear_norm(weight::ScalarLike=1; shape=nothing)
    lambda = _positive(weight, "nuclear-norm weight")
    facts = Property[convex()]
    if shape !== nothing
        dimensions = _shape(shape)
        length(dimensions) == 2 ||
            throw(ArgumentError("a nuclear-norm shape must have two dimensions"))
        push!(facts, lipschitz(lambda * sqrt(minimum(dimensions))))
    end
    return term(
        :nuclear_norm; properties=facts, oracles=_basic_oracles(:value, :subgradient, :prox)
    )
end

"""Declare a linear map for objective composition."""
function linear_operator(
    matrix::AbstractMatrix;
    sigma_max=nothing,
    sigma_min=nothing,
    full_rank_tol::Real=_default_full_rank_tol(),
)
    return term(
        :linear_operator;
        properties=Property[_linear_property(
            matrix; sigma_max=sigma_max, sigma_min=sigma_min, full_rank_tol=full_rank_tol
        )],
        oracles=_basic_oracles(:value, :gradient),
    )
end

"""Declare an affine map for objective composition."""
function affine_operator(
    matrix::AbstractMatrix,
    offset::AbstractVector;
    sigma_max=nothing,
    sigma_min=nothing,
    full_rank_tol::Real=_default_full_rank_tol(),
)
    size(matrix, 1) == length(offset) ||
        throw(DimensionMismatch("matrix rows and offset length must agree"))
    return term(
        :affine_operator;
        properties=Property[_linear_property(
            matrix; sigma_max=sigma_max, sigma_min=sigma_min, full_rank_tol=full_rank_tol
        )],
        oracles=_basic_oracles(:value, :gradient),
    )
end

"""Declare a ridge-regression objective."""
function ridge(
    matrix::AbstractMatrix,
    target::AbstractVector,
    weight::ScalarLike=1;
    sigma_max=nothing,
    sigma_min=nothing,
    full_rank_tol::Real=_default_full_rank_tol(),
)
    size(matrix, 1) == length(target) ||
        throw(DimensionMismatch("matrix rows and target length must agree"))
    mu = _positive(weight, "ridge weight")
    data_min, data_max, _ = _gram_extremes(
        matrix; sigma_max=sigma_max, sigma_min=sigma_min, full_rank_tol=full_rank_tol
    )
    return term(
        :ridge;
        properties=_quadratic_properties(data_min + mu, data_max + mu),
        oracles=_basic_oracles(:value, :gradient),
    )
end

"""Declare the quadratic form `(1/2)x'Qx + q'x`."""
function quadratic_form(
    matrix::AbstractMatrix,
    linear_term::AbstractVector=zeros(size(matrix, 1));
    lambda_min=nothing,
    lambda_max=nothing,
)
    size(matrix, 1) == size(matrix, 2) || throw(ArgumentError("Q must be square"))
    size(matrix, 1) == length(linear_term) ||
        throw(DimensionMismatch("Q and q dimensions must agree"))
    issymmetric(matrix) || throw(ArgumentError("Q must be symmetric"))
    if lambda_min === nothing || lambda_max === nothing
        values = eigvals(Symmetric(Matrix(matrix)))
        lambda_min === nothing && (lambda_min = minimum(values))
        lambda_max === nothing && (lambda_max = maximum(values))
    end
    smallest = scalar(lambda_min)
    largest = scalar(lambda_max)
    return term(
        :quadratic_form;
        properties=_quadratic_properties(smallest, largest),
        oracles=_basic_oracles(:value, :gradient),
    )
end

"""Declare a log-sum-exp loss."""
function log_sum_exp(
    matrix::AbstractMatrix,
    offset::AbstractVector=zeros(size(matrix, 1));
    L=nothing,
    sigma_max=nothing,
)
    size(matrix, 1) == length(offset) ||
        throw(DimensionMismatch("matrix rows and offset length must agree"))
    smoothness = if L === nothing
        largest = if sigma_max === nothing
            first(_compute_singular_extremes(matrix))
        else
            scalar(sigma_max)
        end
        largest^2
    else
        _nonnegative(L, "L")
    end
    return term(
        :log_sum_exp;
        properties=Property[convex(), smooth(smoothness)],
        oracles=_basic_oracles(:value, :gradient),
    )
end

"""Declare a Poisson regression loss without claiming global smoothness."""
function poisson_loss(matrix::AbstractMatrix, counts::AbstractVector)
    size(matrix, 1) == length(counts) ||
        throw(DimensionMismatch("matrix rows and count length must agree"))
    return term(
        :poisson_loss;
        properties=Property[convex()],
        oracles=_basic_oracles(:value, :gradient),
    )
end

"""Declare a summed softplus loss."""
function softplus(matrix::AbstractMatrix; L=nothing, sigma_max=nothing)
    smoothness = if L === nothing
        largest = if sigma_max === nothing
            first(_compute_singular_extremes(matrix))
        else
            scalar(sigma_max)
        end
        largest^2 / 4
    else
        _nonnegative(L, "L")
    end
    return term(
        :softplus;
        properties=Property[convex(), smooth(smoothness)],
        oracles=_basic_oracles(:value, :gradient),
    )
end

"""Declare a weighted infinity norm."""
function linf_norm(weight::ScalarLike=1)
    lambda = _positive(weight, "infinity-norm weight")
    return term(
        :linf_norm;
        properties=Property[convex(), lipschitz(lambda)],
        oracles=_basic_oracles(:value, :subgradient, :prox),
    )
end

"""Declare a group-lasso penalty on a disjoint partition."""
function group_lasso(weight::ScalarLike=1; groups)
    lambda = _positive(weight, "group-lasso weight")
    blocks = [Int[index for index in group] for group in groups]
    isempty(blocks) && throw(ArgumentError("group_lasso requires at least one group"))
    seen = Set{Int}()
    for block in blocks
        isempty(block) && throw(ArgumentError("groups must be nonempty"))
        for index in block
            index > 0 || throw(ArgumentError("group indices must be positive"))
            index in seen && throw(ArgumentError("groups must be disjoint"))
            push!(seen, index)
        end
    end
    return term(
        :group_lasso;
        properties=Property[convex(), lipschitz(lambda * sqrt(length(blocks)))],
        oracles=_basic_oracles(:value, :prox),
    )
end

"""Declare the Tikhonov penalty `(weight/2) * norm(Gamma*x)^2`."""
function tikhonov(
    matrix::AbstractMatrix,
    weight::ScalarLike=1;
    sigma_max=nothing,
    sigma_min=nothing,
    full_rank_tol::Real=_default_full_rank_tol(),
)
    mu = _positive(weight, "Tikhonov weight")
    data_min, data_max, _ = _gram_extremes(
        matrix; sigma_max=sigma_max, sigma_min=sigma_min, full_rank_tol=full_rank_tol
    )
    return term(
        :tikhonov;
        properties=_quadratic_properties(mu * data_min, mu * data_max),
        oracles=_basic_oracles(:value, :gradient, :prox),
    )
end

"""Declare a reverse-Huber penalty."""
function berhu(weight::ScalarLike=1)
    _positive(weight, "berhu weight")
    return term(
        :berhu; properties=Property[convex()], oracles=_basic_oracles(:value, :prox)
    )
end

"""Add a squared Euclidean regularizer through the standard expression algebra."""
function regularize(value::Term; l2::ScalarLike)
    _positive(l2, "regularization weight")
    return value + scalar(l2) * sum_squares()
end

function _required_parameter(values::Vector{Term}, property_name::Symbol, parameter::Symbol)
    result = ScalarExpr[]
    for value in values
        fact = get_property(value, property_name)
        fact === nothing && return nothing
        haskey(fact.parameters, parameter) || return nothing
        push!(result, fact.parameters[parameter])
    end
    return result
end

"""Declare a sum of terms acting on disjoint variable blocks."""
function separable_sum(first::Term, rest::Term...)
    values = Term[first, rest...]
    facts = Property[]
    all(value -> has_property(value, :convex), values) && push!(facts, convex())
    smoothness = _required_parameter(values, :smooth, :L)
    smoothness === nothing || push!(facts, smooth(reduce(max, smoothness)))
    curvature = _required_parameter(values, :strongly_convex, :mu)
    curvature === nothing || push!(facts, strongly_convex(reduce(min, curvature)))
    moduli = _required_parameter(values, :lipschitz, :M)
    moduli === nothing ||
        push!(facts, lipschitz(sqrt(reduce(+, ScalarExpr[value^2 for value in moduli]))))
    diameters = _required_parameter(values, :compact_domain, :diameter)
    diameters === nothing || push!(
        facts,
        compact_domain(sqrt(reduce(+, ScalarExpr[value^2 for value in diameters]))),
    )
    operations = Oracle[]
    for name in (:value, :gradient, :subgradient, :prox, :linear_minimization)
        all(value -> has_oracle(value, name), values) && push!(operations, oracle(name))
    end
    return term(:separable_sum; properties=facts, oracles=operations)
end

end
