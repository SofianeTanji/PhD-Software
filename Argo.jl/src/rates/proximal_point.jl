function proximal_point_bound(; k, initial, γ, kwargs...)
    _require_integer_at_least(k, 1)
    R2 = _require_finite_nonnegative(initial, :initial)
    γf = _require_finite_positive(γ, :γ)
    return R2 / (4 * γf * k)
end

unit_proximal_point_stepsize(; kwargs...) = 1.0

function proximal_point_strongly_convex_rate_factor(; γ, mu, kwargs...)
    γf = _require_finite_positive(γ, :γ)
    muf = _require_finite_positive(mu, :mu)
    return 1 / (1 + γf * muf)
end

function proximal_point_strongly_convex_gap_bound(; k, initial, γ, mu, kwargs...)
    _require_integer_at_least(k, 0)
    gap0 = _require_finite_nonnegative(initial, :initial)
    q = proximal_point_strongly_convex_rate_factor(; γ=γ, mu=mu)
    return q^k * gap0
end

function proximal_point_strongly_convex_distance_bound(; k, initial, γ, mu, kwargs...)
    _require_integer_at_least(k, 0)
    R2 = _require_finite_nonnegative(initial, :initial)
    q = proximal_point_strongly_convex_rate_factor(; γ=γ, mu=mu)
    return q^(2k) * R2
end

function proximal_point_strongly_convex_distance_from_gap_bound(;
    k, initial, γ, mu, kwargs...
)
    muf = _require_finite_positive(mu, :mu)
    return 2 *
           proximal_point_strongly_convex_gap_bound(; k=k, initial=initial, γ=γ, mu=muf) /
           muf
end

function proximal_point_gradient_mapping_bound(; k, initial, γ, kwargs...)
    _require_integer_at_least(k, 0)
    R2 = _require_finite_nonnegative(initial, :initial)
    γf = _require_finite_positive(γ, :γ)
    return R2 / (γf^2 * (k + 1))
end

function guler_accelerated_proximal_point_bound(; k, initial, γ, kwargs...)
    _require_integer_at_least(k, 1)
    R2 = _require_finite_nonnegative(initial, :initial)
    γf = _require_finite_positive(γ, :γ)
    return R2 / (2 * γf * fista_lambda(k)^2)
end

function weakly_convex_proximal_point_stepsize(; rho, kwargs...)
    rhof = _require_finite_positive(rho, :rho)
    return 1 / (2 * rhof)
end

function weakly_convex_proximal_point_gradient_mapping_bound(;
    k, initial, γ, rho, kwargs...
)
    _require_integer_at_least(k, 0)
    gap0 = _require_finite_nonnegative(initial, :initial)
    rhof = _require_finite_positive(rho, :rho)
    γf = _require_finite_positive(γ, :γ)
    known_gamma = _known_real(γf)
    known_rho = _known_real(rhof)
    known_gamma === nothing ||
        known_rho === nothing ||
        known_gamma < 1 / known_rho ||
        throw(DomainError(γ, "requires γ < 1 / rho"))
    return 2 * gap0 / (γf * (1 - γf * rhof) * (k + 1))
end

"""
    proximal_point_bound(; k, initial, γ)

Args: iteration count, initial squared distance, and proximal stepsize.
Returns: scalar bound.
- Evaluates the convex proximal-point gap guarantee.
"""
proximal_point_bound

"""
    unit_proximal_point_stepsize()

Args: none.
Returns: scalar stepsize.
- Gives the unit proximal-point stepsize.
"""
unit_proximal_point_stepsize
