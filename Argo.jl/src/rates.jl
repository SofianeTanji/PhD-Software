"""
Pure convergence-rate formulas and method-parameter schedules.

These functions accept scalar theorem parameters and return scalar bounds or
algorithm parameters. Implementations are grouped by the same families as the
theorem catalogue in `catalogue/theorems/`.
"""

function _require_integer_at_least(k::Integer, lower::Integer, name::Symbol=:k)
    k >= lower || throw(DomainError(k, "$(name) must be >= $(lower)"))
    return k
end

function _require_finite_nonnegative(x::Real, name::Symbol)
    isfinite(x) && x >= 0 || throw(DomainError(x, "$(name) must be finite and nonnegative"))
    return x
end

function _require_finite_nonnegative(x::ScalarExpr, name::Symbol)
    numeric = try_evaluate_scalar(x)
    numeric === nothing || _require_finite_nonnegative(numeric, name)
    return x
end

function _require_finite_positive(x::Real, name::Symbol)
    isfinite(x) && x > 0 || throw(DomainError(x, "$(name) must be finite and positive"))
    return x
end

function _require_finite_positive(x::ScalarExpr, name::Symbol)
    numeric = try_evaluate_scalar(x)
    numeric === nothing || _require_finite_positive(numeric, name)
    return x
end

_known_real(x::Real) = x
_known_real(x::ScalarExpr) = try_evaluate_scalar(x)

function _require_mu_le_L(mu::Real, L::Real)
    muf = _require_finite_positive(mu, :mu)
    Lf = _require_finite_positive(L, :L)
    muf <= Lf || throw(DomainError((mu=mu, L=L), "requires mu <= L"))
    return muf, Lf
end

function _require_mu_lt_L(mu::Real, L::Real)
    muf = _require_finite_positive(mu, :mu)
    Lf = _require_finite_positive(L, :L)
    muf < Lf || throw(DomainError((mu=mu, L=L), "requires mu < L"))
    return muf, Lf
end

function _require_mu_le_L(mu::Number, L::Number)
    muf = _require_finite_positive(mu, :mu)
    Lf = _require_finite_positive(L, :L)
    known_mu = _known_real(muf)
    known_L = _known_real(Lf)
    known_mu === nothing ||
        known_L === nothing ||
        known_mu <= known_L ||
        throw(DomainError((mu=mu, L=L), "requires mu <= L"))
    return muf, Lf
end

function _require_mu_lt_L(mu::Number, L::Number)
    muf = _require_finite_positive(mu, :mu)
    Lf = _require_finite_positive(L, :L)
    known_mu = _known_real(muf)
    known_L = _known_real(Lf)
    known_mu === nothing ||
        known_L === nothing ||
        known_mu < known_L ||
        throw(DomainError((mu=mu, L=L), "requires mu < L"))
    return muf, Lf
end

"""
    fista_lambda(k)

Args: iteration index `k`.
Returns: scalar sequence value.
- Evaluates the Beck-Teboulle/Nesterov lambda sequence with lambda_1 = 1.
"""
function fista_lambda(k::Integer)
    _require_integer_at_least(k, 1)
    λ = 1.0
    for _ in 1:(k - 1)
        λ = (1 + sqrt(1 + 4 * λ^2)) / 2
    end
    return λ
end

"""The standard convex Nesterov/FISTA inertial coefficient at iteration `k`."""
function fista_momentum(; k, kwargs...)
    _require_integer_at_least(k, 0)
    k == 0 && return 0.0
    return (fista_lambda(k) - 1) / fista_lambda(k + 1)
end

"""The FPGM1 coefficient alpha_k = (k - 1) / (k + 2)."""
function simplified_accelerated_momentum(; k, kwargs...)
    _require_integer_at_least(k, 0)
    k == 0 && return 0.0
    return (k - 1) / (k + 2)
end

"""The fixed inertial coefficient for smooth strongly convex acceleration."""
function strongly_convex_accelerated_momentum(; L, mu, kwargs...)
    muf, Lf = _require_mu_le_L(mu, L)
    root_condition = sqrt(muf / Lf)
    return (1 - root_condition) / (1 + root_condition)
end

include("rates/conditional_gradient.jl")
include("rates/gradient.jl")
include("rates/inexact.jl")
include("rates/projected.jl")
include("rates/proximal_gradient.jl")
include("rates/proximal_point.jl")
include("rates/splitting.jl")
include("rates/subgradient.jl")
