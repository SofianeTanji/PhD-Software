function proximal_gradient_convex_bound(; k, initial, L, kwargs...)
    _require_integer_at_least(k, 1)
    R2 = _require_finite_nonnegative(initial, :initial)
    Lf = _require_finite_positive(L, :L)
    return Lf * R2 / (2 * k)
end

proximal_gradient_step(; L, kwargs...) = 1 / _require_finite_positive(L, :L)

function proximal_gradient_average_bound(; k, initial, L, kwargs...)
    _require_integer_at_least(k, 0)
    R2 = _require_finite_nonnegative(initial, :initial)
    Lf = _require_finite_positive(L, :L)
    return Lf * R2 / (2 * (k + 1))
end

# Ergodic (averaged) FISTA. Beck-Teboulle Thm. 4.4 bounds the LAST iterate,
# F(x_j) − F* ≤ 2L R²/(j+1)². The average only inherits the summed bound:
#   F(x̄_k) − F* ≤ (1/k) Σ_j 2L R²/(j+1)² ≤ 2(π²/6 − 1) L R²/(k+1),
# since Σ_{j≥1} 1/(j+1)² = π²/6 − 1. So the ergodic average is O(1/k), not
# O(1/k²) — acceleration does not survive averaging.
const _FISTA_ERGODIC_CONSTANT = 2 * (pi^2 / 6 - 1)

function fista_composite_average_bound(; k, initial, L, kwargs...)
    _require_integer_at_least(k, 0)
    R2 = _require_finite_nonnegative(initial, :initial)
    Lf = _require_finite_positive(L, :L)
    return _FISTA_ERGODIC_CONSTANT * Lf * R2 / (k + 1)
end

function fista_composite_convex_bound(; k, initial, L, kwargs...)
    R2 = _require_finite_nonnegative(initial, :initial)
    Lf = _require_finite_positive(L, :L)
    return Lf * R2 / (2 * fista_lambda(k)^2)
end

function simplified_accelerated_proximal_gradient_bound(; k, initial, L, kwargs...)
    _require_integer_at_least(k, 0)
    R2 = _require_finite_nonnegative(initial, :initial)
    Lf = _require_finite_positive(L, :L)
    return 2 * Lf * R2 / (k^2 + 5k + 2)
end

function proximal_gradient_strongly_convex_gap_bound(; k, initial, L, mu, kwargs...)
    return gd_smooth_strongly_convex_gap_bound(; k=k, initial=initial, L=L, mu=mu)
end

function sc_fista_composite_strongly_convex_bound(; k, initial, L, mu, kwargs...)
    _require_integer_at_least(k, 0)
    R2 = _require_finite_nonnegative(initial, :initial)
    muf, Lf = _require_mu_lt_L(mu, L)
    q = muf / Lf
    return (Lf / 2) * (1 + sqrt(q)) * (1 - sqrt(q))^k * R2
end

function sc_fista_composite_strongly_convex_lyapunov_bound(; k, initial, L, mu, kwargs...)
    _require_integer_at_least(k, 0)
    energy = _require_finite_nonnegative(initial, :initial)
    muf, Lf = _require_mu_lt_L(mu, L)
    return (1 - sqrt(muf / Lf))^k * energy
end

function sc_fista_composite_gap_from_gap_bound(; k, initial, L, mu, kwargs...)
    gap0 = _require_finite_nonnegative(initial, :initial)
    return 2 * sc_fista_composite_strongly_convex_lyapunov_bound(;
        k=k, initial=gap0, L=L, mu=mu
    )
end

function sc_fista_composite_distance_from_distance_bound(; k, initial, L, mu, kwargs...)
    muf = _require_finite_positive(mu, :mu)
    return 2 *
           sc_fista_composite_strongly_convex_bound(; k=k, initial=initial, L=L, mu=muf) /
           muf
end

function sc_fista_composite_distance_from_lyapunov_bound(; k, initial, L, mu, kwargs...)
    muf = _require_finite_positive(mu, :mu)
    return 2 * sc_fista_composite_strongly_convex_lyapunov_bound(;
        k=k, initial=initial, L=L, mu=muf
    ) / muf
end

function sc_fista_composite_gradient_mapping_from_distance_bound(;
    k, initial, L, mu, kwargs...
)
    Lf = _require_finite_positive(L, :L)
    return 2 *
           Lf *
           sc_fista_composite_strongly_convex_bound(; k=k, initial=initial, L=Lf, mu=mu)
end

function sc_fista_composite_gradient_mapping_from_lyapunov_bound(;
    k, initial, L, mu, kwargs...
)
    Lf = _require_finite_positive(L, :L)
    return 2 *
           Lf *
           sc_fista_composite_strongly_convex_lyapunov_bound(;
               k=k, initial=initial, L=Lf, mu=mu
           )
end

sc_fista_strong_convexity_ratio(; L, mu, kwargs...) = begin
    muf, Lf = _require_mu_lt_L(mu, L)
    muf / Lf
end

sc_fista_linear_factor(; L, mu, kwargs...) = begin
    muf, Lf = _require_mu_lt_L(mu, L)
    1 - sqrt(muf / Lf)
end

function proximal_gradient_gradient_mapping_bound(; k, initial, L, kwargs...)
    _require_integer_at_least(k, 0)
    gap0 = _require_finite_nonnegative(initial, :initial)
    Lf = _require_finite_positive(L, :L)
    return 2 * Lf * gap0 / (k + 1)
end

function proximal_gradient_mapping_from_distance_bound(; k, initial, L, kwargs...)
    Lf = _require_finite_positive(L, :L)
    return 2 * Lf * proximal_gradient_convex_bound(; k=k, initial=initial, L=Lf)
end

function proximal_gradient_average_mapping_from_distance_bound(; k, initial, L, kwargs...)
    Lf = _require_finite_positive(L, :L)
    return 2 * Lf * proximal_gradient_average_bound(; k=k, initial=initial, L=Lf)
end

function fista_composite_mapping_from_distance_bound(; k, initial, L, kwargs...)
    Lf = _require_finite_positive(L, :L)
    return 2 * Lf * fista_composite_convex_bound(; k=k, initial=initial, L=Lf)
end

function fista_composite_average_mapping_from_distance_bound(; k, initial, L, kwargs...)
    Lf = _require_finite_positive(L, :L)
    return 2 * Lf * fista_composite_average_bound(; k=k, initial=initial, L=Lf)
end

function simplified_accelerated_proximal_gradient_mapping_from_distance_bound(;
    k, initial, L, kwargs...
)
    Lf = _require_finite_positive(L, :L)
    return 2 *
           Lf *
           simplified_accelerated_proximal_gradient_bound(; k=k, initial=initial, L=Lf)
end

proximal_gradient_nonconvex_step(; L, kwargs...) = 1 / (2 * _require_finite_positive(L, :L))

function proximal_gradient_nonconvex_mapping_bound(;
    k, initial, L, step_size=nothing, kwargs...
)
    _require_integer_at_least(k, 0)
    gap0 = _require_finite_nonnegative(initial, :initial)
    Lf = _require_finite_positive(L, :L)
    γ = if step_size === nothing
        proximal_gradient_nonconvex_step(; L=Lf)
    else
        _require_finite_positive(step_size, :step_size)
    end
    known_gamma = _known_real(γ)
    known_L = _known_real(Lf)
    known_gamma === nothing ||
        known_L === nothing ||
        known_gamma < 1 / known_L ||
        throw(DomainError(γ, "requires step_size < 1 / L"))
    return 2 * gap0 / (γ * (1 - γ * Lf) * (k + 1))
end

function proximal_gradient_strongly_convex_distance_bound(; k, initial, L, mu, kwargs...)
    _require_integer_at_least(k, 1)
    R2 = _require_finite_nonnegative(initial, :initial)
    muf, Lf = _require_mu_le_L(mu, L)
    q = 1 - muf / Lf
    return (Lf / 2) * q^(k - 1) * R2
end

function proximal_gradient_g_strongly_convex_rate_factor(; L, mu, kwargs...)
    muf = _require_finite_positive(mu, :mu)
    Lf = _require_finite_positive(L, :L)
    return Lf / (Lf + muf)
end

function proximal_gradient_g_strongly_convex_gap_bound(; k, initial, L, mu, kwargs...)
    _require_integer_at_least(k, 0)
    gap0 = _require_finite_nonnegative(initial, :initial)
    q = proximal_gradient_g_strongly_convex_rate_factor(; L=L, mu=mu)
    return q^k * gap0
end

function proximal_gradient_g_strongly_convex_distance_bound(; k, initial, L, mu, kwargs...)
    _require_integer_at_least(k, 1)
    R2 = _require_finite_nonnegative(initial, :initial)
    Lf = _require_finite_positive(L, :L)
    q = proximal_gradient_g_strongly_convex_rate_factor(; L=Lf, mu=mu)
    return (Lf / 2) * q^(k - 1) * R2
end

function proximal_gradient_sc_gradient_mapping_gap_bound(; k, initial, L, mu, kwargs...)
    Lf = _require_finite_positive(L, :L)
    return 2 *
           Lf *
           proximal_gradient_strongly_convex_gap_bound(; k=k, initial=initial, L=Lf, mu=mu)
end

function proximal_gradient_sc_gradient_mapping_distance_bound(;
    k, initial, L, mu, kwargs...
)
    Lf = _require_finite_positive(L, :L)
    return 2 *
           Lf *
           proximal_gradient_strongly_convex_distance_bound(;
               k=k, initial=initial, L=Lf, mu=mu
           )
end

function proximal_gradient_g_sc_gradient_mapping_gap_bound(; k, initial, L, mu, kwargs...)
    Lf = _require_finite_positive(L, :L)
    return 2 *
           Lf *
           proximal_gradient_g_strongly_convex_gap_bound(;
               k=k, initial=initial, L=Lf, mu=mu
           )
end

function proximal_gradient_g_sc_gradient_mapping_distance_bound(;
    k, initial, L, mu, kwargs...
)
    Lf = _require_finite_positive(L, :L)
    return 2 *
           Lf *
           proximal_gradient_g_strongly_convex_distance_bound(;
               k=k, initial=initial, L=Lf, mu=mu
           )
end

function proximal_gradient_sc_distance_from_gap_bound(; k, initial, L, mu, kwargs...)
    muf = _require_finite_positive(mu, :mu)
    return 2 * proximal_gradient_strongly_convex_gap_bound(;
        k=k, initial=initial, L=L, mu=muf
    ) / muf
end

function proximal_gradient_sc_distance_from_distance_bound(; k, initial, L, mu, kwargs...)
    muf = _require_finite_positive(mu, :mu)
    return 2 * proximal_gradient_strongly_convex_distance_bound(;
        k=k, initial=initial, L=L, mu=muf
    ) / muf
end

function proximal_gradient_g_sc_distance_from_gap_bound(; k, initial, L, mu, kwargs...)
    muf = _require_finite_positive(mu, :mu)
    return 2 * proximal_gradient_g_strongly_convex_gap_bound(;
        k=k, initial=initial, L=L, mu=muf
    ) / muf
end

function proximal_gradient_g_sc_distance_from_distance_bound(; k, initial, L, mu, kwargs...)
    muf = _require_finite_positive(mu, :mu)
    return 2 * proximal_gradient_g_strongly_convex_distance_bound(;
        k=k, initial=initial, L=L, mu=muf
    ) / muf
end

# These guarantees describe gradient-mapping descent. Both the proximal-gradient
# and projected catalogue families reference them.
function proximal_gradient_gmd_gap_bound(; k, initial, L, mu, kwargs...)
    _require_integer_at_least(k, 0)
    gap0 = _require_finite_nonnegative(initial, :initial)
    muf, Lf = _require_mu_le_L(mu, L)
    return (1 - muf / Lf)^k * gap0
end

function proximal_gradient_gmd_mapping_bound(; k, initial, L, mu, kwargs...)
    Lf = _require_finite_positive(L, :L)
    return 2 * Lf * proximal_gradient_gmd_gap_bound(; k=k, initial=initial, L=Lf, mu=mu)
end

"""
    proximal_gradient_convex_bound(; k, initial, L)

Args: iteration count, initial squared distance, and smoothness.
Returns: scalar bound.
- Evaluates the convex proximal-gradient gap guarantee.
"""
proximal_gradient_convex_bound

"""
    proximal_gradient_step(; L)

Args: smoothness constant.
Returns: scalar stepsize.
- Gives the fixed proximal-gradient stepsize.
"""
proximal_gradient_step

"""
    proximal_gradient_average_bound(; k, initial, L)

Args: iteration count, initial squared distance, and smoothness.
Returns: scalar bound.
- Evaluates the averaged convex proximal-gradient gap guarantee.
"""
proximal_gradient_average_bound

"""
    fista_composite_average_bound(; k, initial, L)

Args: iteration count, initial squared distance, and smoothness.
Returns: scalar bound.
- Evaluates the averaged composite FISTA gap guarantee.
"""
fista_composite_average_bound

"""
    fista_composite_convex_bound(; k, initial, L)

Args: iteration count, initial squared distance, and smoothness.
Returns: scalar bound.
- Evaluates the composite FISTA convex gap guarantee.
"""
fista_composite_convex_bound

"""
    simplified_accelerated_proximal_gradient_bound(; k, initial, L)

Args: iteration count, initial squared distance, and smoothness.
Returns: scalar bound.
- Evaluates a simplified accelerated proximal-gradient guarantee.
"""
simplified_accelerated_proximal_gradient_bound

"""
    proximal_gradient_strongly_convex_gap_bound(; k, initial, L, mu)

Args: iteration count, initial gap, smoothness, and curvature.
Returns: scalar bound.
- Evaluates the strongly convex proximal-gradient gap guarantee.
"""
proximal_gradient_strongly_convex_gap_bound

"""
    sc_fista_composite_strongly_convex_bound(; k, initial, L, mu)

Args: iteration count, initial squared distance, smoothness, and curvature.
Returns: scalar bound.
- Evaluates the strongly convex composite FISTA distance guarantee.
"""
sc_fista_composite_strongly_convex_bound

"""
    sc_fista_composite_strongly_convex_lyapunov_bound(; k, initial, L, mu)

Args: iteration count, initial Lyapunov value, smoothness, and curvature.
Returns: scalar bound.
- Evaluates the strongly convex composite FISTA Lyapunov guarantee.
"""
sc_fista_composite_strongly_convex_lyapunov_bound

"""
    sc_fista_composite_gap_from_gap_bound(; k, initial, L, mu)

Args: iteration count, initial objective gap, smoothness, and strong convexity.
Returns: SC-FISTA objective-gap bound after the explicit strong-convexity
gap-to-Lyapunov conversion `E0 <= 2 * initial`.
"""
sc_fista_composite_gap_from_gap_bound

"""
    sc_fista_strong_convexity_ratio(; L, mu)

Args: smoothness and strong convexity constants.
Returns: scalar ratio.
- Computes the curvature ratio used by strongly convex FISTA.
"""
sc_fista_strong_convexity_ratio

"""
    sc_fista_linear_factor(; L, mu)

Args: smoothness and strong convexity constants.
Returns: scalar factor.
- Computes the linear contraction factor used by strongly convex FISTA.
"""
sc_fista_linear_factor
