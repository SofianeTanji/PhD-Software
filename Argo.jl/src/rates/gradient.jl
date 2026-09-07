function gd_smooth_convex_bound(; k, initial, L, kwargs...)
    _require_integer_at_least(k, 0)
    R2 = _require_finite_nonnegative(initial, :initial)
    Lf = _require_finite_positive(L, :L)
    return Lf * R2 / (2 * (2k + 1))
end

gd_smooth_convex_step(; L, kwargs...) = 1 / _require_finite_positive(L, :L)

rotaru2026_stepsize_upper(; L, kwargs...) = 2 / _require_finite_positive(L, :L)

function rotaru2026_smooth_convex_gradient_norm_bound(;
    k, initial, L, step_size, kwargs...
)
    _require_integer_at_least(k, 0)
    gap0 = _require_finite_nonnegative(initial, :initial)
    Lf = _require_finite_positive(L, :L)
    γf = _require_finite_positive(step_size, :step_size)
    normalized_step = γf * Lf
    known_step = _known_real(normalized_step)
    known_step === nothing || known_step < 2 ||
        throw(DomainError(step_size, "requires step_size * L < 2"))
    growth = (1 - normalized_step)^(-2k) - 1
    denominator = 1 + min(2k * normalized_step, growth)
    return 2 * Lf * gap0 / denominator
end

function accelerated_gradient_smooth_convex_bound(; k, initial, L, kwargs...)
    R2 = _require_finite_nonnegative(initial, :initial)
    Lf = _require_finite_positive(L, :L)
    return Lf * R2 / (2 * fista_lambda(k)^2)
end

function simplified_accelerated_gradient_smooth_convex_bound(; k, initial, L, kwargs...)
    _require_integer_at_least(k, 0)
    R2 = _require_finite_nonnegative(initial, :initial)
    Lf = _require_finite_positive(L, :L)
    return 2 * Lf * R2 / (k^2 + 5k + 6)
end

function gd_smooth_strongly_convex_distance_bound(; k, initial, L, mu, kwargs...)
    _require_integer_at_least(k, 0)
    R2 = _require_finite_nonnegative(initial, :initial)
    muf, Lf = _require_mu_le_L(mu, L)
    return (Lf / 2) * ((Lf - muf) / (Lf + muf))^(2k) * R2
end

function gd_smooth_strongly_convex_optimal_step(; L, mu, kwargs...)
    muf, Lf = _require_mu_le_L(mu, L)
    return 2 / (Lf + muf)
end

function accelerated_gradient_smooth_strongly_convex_distance_bound(;
    k, initial, L, mu, kwargs...
)
    _require_integer_at_least(k, 0)
    R2 = _require_finite_nonnegative(initial, :initial)
    muf, Lf = _require_mu_le_L(mu, L)
    return ((Lf + muf) / 2) * (1 - sqrt(muf / Lf))^k * R2
end

function gd_smooth_strongly_convex_gap_bound(; k, initial, L, mu, kwargs...)
    _require_integer_at_least(k, 0)
    gap0 = _require_finite_nonnegative(initial, :initial)
    muf, Lf = _require_mu_lt_L(mu, L)
    return (1 - muf / Lf)^k * gap0
end

linear_rate_factor(; L, mu, kwargs...) = begin
    muf, Lf = _require_mu_lt_L(mu, L)
    1 - muf / Lf
end

function gd_smooth_convex_gradient_norm_from_distance_bound(; k, initial, L, kwargs...)
    _require_integer_at_least(k, 0)
    R2 = _require_finite_nonnegative(initial, :initial)
    Lf = _require_finite_positive(L, :L)
    return Lf^2 * R2 / (2k + 1)
end

function accelerated_gradient_smooth_convex_gradient_norm_from_distance_bound(;
    k, initial, L, kwargs...
)
    R2 = _require_finite_nonnegative(initial, :initial)
    Lf = _require_finite_positive(L, :L)
    return Lf^2 * R2 / fista_lambda(k)^2
end

function simplified_accelerated_gradient_smooth_convex_gradient_norm_from_distance_bound(;
    k, initial, L, kwargs...
)
    _require_integer_at_least(k, 0)
    R2 = _require_finite_nonnegative(initial, :initial)
    Lf = _require_finite_positive(L, :L)
    return 4 * Lf^2 * R2 / (k^2 + 5k + 6)
end

function gd_smooth_strongly_convex_gradient_norm_from_gap_bound(;
    k, initial, L, mu, kwargs...
)
    Lf = _require_finite_positive(L, :L)
    return 2 * Lf * gd_smooth_strongly_convex_gap_bound(; k=k, initial=initial, L=Lf, mu=mu)
end

function gd_smooth_strongly_convex_gradient_norm_from_distance_bound(;
    k, initial, L, mu, kwargs...
)
    Lf = _require_finite_positive(L, :L)
    return 2 *
           Lf *
           gd_smooth_strongly_convex_distance_bound(; k=k, initial=initial, L=Lf, mu=mu)
end

function accelerated_gradient_smooth_strongly_convex_gradient_norm_from_distance_bound(;
    k, initial, L, mu, kwargs...
)
    Lf = _require_finite_positive(L, :L)
    return 2 *
           Lf *
           accelerated_gradient_smooth_strongly_convex_distance_bound(;
               k=k, initial=initial, L=Lf, mu=mu
           )
end

function accelerated_gradient_smooth_strongly_convex_gradient_norm_from_lyapunov_bound(;
    k, initial, L, mu, kwargs...
)
    Lf = _require_finite_positive(L, :L)
    return 2 *
           Lf *
           accelerated_gradient_smooth_strongly_convex_lyapunov_bound(;
               k=k, initial=initial, L=Lf, mu=mu
           )
end

function gd_smooth_strongly_convex_distance_from_gap_bound(; k, initial, L, mu, kwargs...)
    muf = _require_finite_positive(mu, :mu)
    return 2 * gd_smooth_strongly_convex_gap_bound(; k=k, initial=initial, L=L, mu=muf) /
           muf
end

function gd_smooth_strongly_convex_distance_contraction_bound(;
    k, initial, L, mu, kwargs...
)
    _require_integer_at_least(k, 0)
    R2 = _require_finite_nonnegative(initial, :initial)
    muf, Lf = _require_mu_le_L(mu, L)
    return ((Lf - muf) / (Lf + muf))^(2k) * R2
end

function accelerated_gradient_smooth_strongly_convex_distance_from_distance_bound(;
    k, initial, L, mu, kwargs...
)
    _require_integer_at_least(k, 0)
    R2 = _require_finite_nonnegative(initial, :initial)
    muf, Lf = _require_mu_le_L(mu, L)
    return ((Lf + muf) / muf) * (1 - sqrt(muf / Lf))^k * R2
end

function accelerated_gradient_smooth_strongly_convex_distance_from_lyapunov_bound(;
    k, initial, L, mu, kwargs...
)
    muf = _require_finite_positive(mu, :mu)
    return 2 * accelerated_gradient_smooth_strongly_convex_lyapunov_bound(;
        k=k, initial=initial, L=L, mu=muf
    ) / muf
end

function gd_smooth_pl_gap_bound(; k, initial, L, mu, kwargs...)
    _require_integer_at_least(k, 0)
    gap0 = _require_finite_nonnegative(initial, :initial)
    muf, Lf = _require_mu_le_L(mu, L)
    return (1 - muf / Lf)^k * gap0
end

function gd_smooth_pl_gradient_norm_squared_bound(; k, initial, L, mu, kwargs...)
    _require_integer_at_least(k, 0)
    gap0 = _require_finite_nonnegative(initial, :initial)
    muf, Lf = _require_mu_le_L(mu, L)
    return 2 * Lf * (1 - muf / Lf)^k * gap0
end

gd_smooth_pl_step(; L, kwargs...) = gd_smooth_convex_step(; L=L)

pl_linear_rate_factor(; L, mu, kwargs...) = begin
    muf, Lf = _require_mu_le_L(mu, L)
    1 - muf / Lf
end

function gd_smooth_pl_gap_from_distance_bound(; k, initial, L, mu, kwargs...)
    _require_integer_at_least(k, 0)
    R2 = _require_finite_nonnegative(initial, :initial)
    muf, Lf = _require_mu_le_L(mu, L)
    return (Lf / 2) * (1 - muf / Lf)^k * R2
end

function gd_smooth_pl_gradient_norm_from_distance_bound(; k, initial, L, mu, kwargs...)
    _require_integer_at_least(k, 0)
    R2 = _require_finite_nonnegative(initial, :initial)
    muf, Lf = _require_mu_le_L(mu, L)
    return Lf^2 * (1 - muf / Lf)^k * R2
end

function gd_smooth_convex_gradient_mapping_bound(; k, initial, L, kwargs...)
    _require_integer_at_least(k, 0)
    gap0 = _require_finite_nonnegative(initial, :initial)
    Lf = _require_finite_positive(L, :L)
    return 2 * Lf * gap0 / (k + 1)
end

function gd_smooth_nonconvex_gradient_mapping_bound(; kwargs...)
    gd_smooth_convex_gradient_mapping_bound(; kwargs...)
end

function accelerated_gradient_smooth_strongly_convex_lyapunov_bound(; kwargs...)
    sc_fista_composite_strongly_convex_lyapunov_bound(; kwargs...)
end

function quadratic_gd_spectral_factor(; L, mu, kwargs...)
    muf, Lf = _require_mu_le_L(mu, L)
    return (1 - muf / Lf)^2
end

function quadratic_gd_spectral_distance_bound(; k, initial, L, mu, kwargs...)
    _require_integer_at_least(k, 0)
    R2 = _require_finite_nonnegative(initial, :initial)
    return quadratic_gd_spectral_factor(; L=L, mu=mu)^k * R2
end

function quadratic_gd_spectral_gap_from_distance_bound(; k, initial, L, mu, kwargs...)
    Lf = _require_finite_positive(L, :L)
    return (Lf / 2) *
           quadratic_gd_spectral_distance_bound(; k=k, initial=initial, L=Lf, mu=mu)
end

function quadratic_gd_spectral_gap_from_gap_bound(; k, initial, L, mu, kwargs...)
    _require_integer_at_least(k, 0)
    gap0 = _require_finite_nonnegative(initial, :initial)
    return quadratic_gd_spectral_factor(; L=L, mu=mu)^k * gap0
end

function quadratic_gd_spectral_distance_from_gap_bound(; k, initial, L, mu, kwargs...)
    muf = _require_finite_positive(mu, :mu)
    return 2 *
           quadratic_gd_spectral_gap_from_gap_bound(; k=k, initial=initial, L=L, mu=muf) /
           muf
end

function quadratic_gd_spectral_gradient_norm_from_distance_bound(;
    k, initial, L, mu, kwargs...
)
    Lf = _require_finite_positive(L, :L)
    return Lf^2 * quadratic_gd_spectral_distance_bound(; k=k, initial=initial, L=Lf, mu=mu)
end

function quadratic_gd_spectral_gradient_norm_from_gap_bound(; k, initial, L, mu, kwargs...)
    Lf = _require_finite_positive(L, :L)
    return 2 *
           Lf *
           quadratic_gd_spectral_gap_from_gap_bound(; k=k, initial=initial, L=Lf, mu=mu)
end

"""
    gd_smooth_convex_bound(; k, initial, L)

Args: iteration count, initial squared distance, and smoothness.
Returns: scalar bound.
- Evaluates the smooth convex gradient-descent gap guarantee.
"""
gd_smooth_convex_bound

"""
    gd_smooth_convex_step(; L)

Args: smoothness constant.
Returns: scalar stepsize.
- Gives the fixed gradient-descent stepsize for smooth convex objectives.
"""
gd_smooth_convex_step

"""
    rotaru2026_stepsize_upper(; L)

Args: smoothness constant.
Returns: strict upper endpoint of the admissible fixed-stepsize interval.
"""
rotaru2026_stepsize_upper

"""
    rotaru2026_smooth_convex_gradient_norm_bound(; k, initial, L, step_size)

Args: iteration count, initial objective gap, smoothness, and fixed stepsize.
Returns: last-iterate squared gradient-norm bound for `0 < step_size * L < 2`.
"""
rotaru2026_smooth_convex_gradient_norm_bound

"""
    accelerated_gradient_smooth_convex_bound(; k, initial, L)

Args: iteration count, initial squared distance, and smoothness.
Returns: scalar bound.
- Evaluates the accelerated smooth convex gap guarantee.
"""
accelerated_gradient_smooth_convex_bound

"""
    simplified_accelerated_gradient_smooth_convex_bound(; k, initial, L)

Args: iteration count, initial squared distance, and smoothness.
Returns: scalar bound.
- Evaluates a simplified accelerated smooth convex gap guarantee.
"""
simplified_accelerated_gradient_smooth_convex_bound

"""
    gd_smooth_strongly_convex_distance_bound(; k, initial, L, mu)

Args: iteration count, initial squared distance, smoothness, and curvature.
Returns: scalar bound.
- Evaluates the optimal-step gradient-descent distance guarantee.
"""
gd_smooth_strongly_convex_distance_bound

"""
    gd_smooth_strongly_convex_optimal_step(; L, mu)

Args: smoothness and strong convexity constants.
Returns: scalar stepsize.
- Gives the optimal fixed gradient-descent stepsize for smooth strong convexity.
"""
gd_smooth_strongly_convex_optimal_step

"""
    accelerated_gradient_smooth_strongly_convex_distance_bound(; k, initial, L, mu)

Args: iteration count, initial squared distance, smoothness, and curvature.
Returns: scalar bound.
- Evaluates the accelerated smooth strongly convex distance guarantee.
"""
accelerated_gradient_smooth_strongly_convex_distance_bound

"""
    gd_smooth_strongly_convex_gap_bound(; k, initial, L, mu)

Args: iteration count, initial gap, smoothness, and curvature.
Returns: scalar bound.
- Evaluates the gradient-descent linear objective-gap guarantee.
"""
gd_smooth_strongly_convex_gap_bound

"""
    linear_rate_factor(; L, mu)

Args: smoothness and strong convexity constants.
Returns: scalar factor.
- Computes the contraction factor for a linear rate.
"""
linear_rate_factor
