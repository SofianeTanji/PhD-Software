function projected_gradient_convex_compact_bound(; k, L, diameter, kwargs...)
    _require_integer_at_least(k, 1)
    Lf = _require_finite_positive(L, :L)
    D = _require_finite_nonnegative(diameter, :diameter)
    return Lf * D^2 / (2 * k)
end

function projected_gradient_convex_compact_average_bound(; k, L, diameter, kwargs...)
    _require_integer_at_least(k, 0)
    Lf = _require_finite_positive(L, :L)
    D = _require_finite_nonnegative(diameter, :diameter)
    return Lf * D^2 / (2 * (k + 1))
end

function accelerated_projected_gradient_convex_compact_bound(; k, L, diameter, kwargs...)
    Lf = _require_finite_positive(L, :L)
    D = _require_finite_nonnegative(diameter, :diameter)
    return Lf * D^2 / (2 * fista_lambda(k)^2)
end

function simplified_accelerated_projected_gradient_compact_bound(;
    k, L, diameter, kwargs...
)
    _require_integer_at_least(k, 0)
    Lf = _require_finite_positive(L, :L)
    D = _require_finite_nonnegative(diameter, :diameter)
    return 2 * Lf * D^2 / (k^2 + 5k + 2)
end

function projected_subgradient_convex_lipschitz_compact_bound(; k, M, diameter, kwargs...)
    _require_integer_at_least(k, 0)
    Mf = _require_finite_positive(M, :M)
    D = _require_finite_nonnegative(diameter, :diameter)
    return Mf * D / sqrt(k + 1)
end

function projected_subgradient_compact_step(; k, M, diameter, kwargs...)
    _require_integer_at_least(k, 0)
    Mf = _require_finite_positive(M, :M)
    D = _require_finite_nonnegative(diameter, :diameter)
    return D / (Mf * sqrt(k + 1))
end

function projected_gradient_compact_mapping_bound(; k, L, diameter, kwargs...)
    Lf = _require_finite_positive(L, :L)
    return 2 * Lf * projected_gradient_convex_compact_bound(; k=k, L=Lf, diameter=diameter)
end

function projected_gradient_compact_average_mapping_bound(; k, L, diameter, kwargs...)
    Lf = _require_finite_positive(L, :L)
    return 2 *
           Lf *
           projected_gradient_convex_compact_average_bound(; k=k, L=Lf, diameter=diameter)
end

function accelerated_projected_gradient_compact_mapping_bound(; k, L, diameter, kwargs...)
    Lf = _require_finite_positive(L, :L)
    return 2 *
           Lf *
           accelerated_projected_gradient_convex_compact_bound(;
               k=k, L=Lf, diameter=diameter
           )
end

function simplified_accelerated_projected_gradient_compact_mapping_bound(;
    k, L, diameter, kwargs...
)
    Lf = _require_finite_positive(L, :L)
    return 2 *
           Lf *
           simplified_accelerated_projected_gradient_compact_bound(;
               k=k, L=Lf, diameter=diameter
           )
end

function projected_gradient_sc_compact_gap_bound(; k, L, mu, diameter, kwargs...)
    D = _require_finite_nonnegative(diameter, :diameter)
    return proximal_gradient_strongly_convex_distance_bound(; k=k, initial=D^2, L=L, mu=mu)
end

function projected_gradient_sc_compact_mapping_bound(; k, L, mu, diameter, kwargs...)
    Lf = _require_finite_positive(L, :L)
    return 2 *
           Lf *
           projected_gradient_sc_compact_gap_bound(; k=k, L=Lf, mu=mu, diameter=diameter)
end

function projected_gradient_sc_compact_distance_bound(; k, L, mu, diameter, kwargs...)
    muf = _require_finite_positive(mu, :mu)
    return 2 *
           projected_gradient_sc_compact_gap_bound(; k=k, L=L, mu=muf, diameter=diameter) /
           muf
end
