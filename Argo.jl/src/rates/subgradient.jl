function subgradient_convex_lipschitz_bound(; k, initial, M, kwargs...)
    _require_integer_at_least(k, 0)
    R2 = _require_finite_nonnegative(initial, :initial)
    Mf = _require_finite_positive(M, :M)
    return Mf * sqrt(R2) / sqrt(k + 1)
end

function subgradient_convex_lipschitz_step(; M, initial, k, kwargs...)
    _require_integer_at_least(k, 0)
    R2 = _require_finite_nonnegative(initial, :initial)
    Mf = _require_finite_positive(M, :M)
    return sqrt(R2) / (Mf * sqrt(k + 1))
end

function proximal_subgradient_convex_lipschitz_bound(; kwargs...)
    subgradient_convex_lipschitz_bound(; kwargs...)
end

function proximal_subgradient_convex_lipschitz_step(; kwargs...)
    subgradient_convex_lipschitz_step(; kwargs...)
end

function subgradient_strongly_convex_lipschitz_bound(; k, initial=nothing, M, mu, kwargs...)
    _require_integer_at_least(k, 0)
    Mf = _require_finite_positive(M, :M)
    muf = _require_finite_positive(mu, :mu)
    # Conservative weighted-average bound for Lipschitz, mu-strongly convex objectives.
    return 2 * Mf^2 / (muf * (k + 1))
end

function subgradient_strongly_convex_lipschitz_step(; k, mu, kwargs...)
    _require_integer_at_least(k, 0)
    muf = _require_finite_positive(mu, :mu)
    return 2 / (muf * (k + 1))
end

function proximal_subgradient_strongly_convex_lipschitz_bound(; kwargs...)
    subgradient_strongly_convex_lipschitz_bound(; kwargs...)
end

function proximal_subgradient_strongly_convex_lipschitz_step(; kwargs...)
    subgradient_strongly_convex_lipschitz_step(; kwargs...)
end

function subgradient_strongly_convex_lipschitz_distance_bound(; k, M, mu, kwargs...)
    muf = _require_finite_positive(mu, :mu)
    return 2 * subgradient_strongly_convex_lipschitz_bound(; k=k, M=M, mu=muf, kwargs...) /
           muf
end

function proximal_subgradient_strongly_convex_lipschitz_distance_bound(; kwargs...)
    subgradient_strongly_convex_lipschitz_distance_bound(; kwargs...)
end

"""
    subgradient_convex_lipschitz_bound(; k, initial, M)

Args: iteration count, initial squared distance, and Lipschitz constant.
Returns: scalar bound.
- Evaluates the convex Lipschitz subgradient gap guarantee.
"""
subgradient_convex_lipschitz_bound

"""
    subgradient_convex_lipschitz_step(; M, initial, k)

Args: Lipschitz constant, initial squared distance, and iteration count.
Returns: scalar stepsize.
- Gives the diminishing subgradient stepsize for convex Lipschitz objectives.
"""
subgradient_convex_lipschitz_step
