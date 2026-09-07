function frank_wolfe_convex_diameter_bound(; k, L, diameter, kwargs...)
    _require_integer_at_least(k, 1)
    Lf = _require_finite_nonnegative(L, :L)
    D = _require_finite_nonnegative(diameter, :diameter)
    return 2 * Lf * D^2 / (k + 2)
end

function frank_wolfe_open_loop_step(; k, kwargs...)
    _require_integer_at_least(k, 0)
    return 2 / (k + 2)
end

function frank_wolfe_nonconvex_gap_bound(; k, initial, L, diameter, kwargs...)
    _require_integer_at_least(k, 0)
    gap0 = _require_finite_nonnegative(initial, :initial)
    Lf = _require_finite_nonnegative(L, :L)
    D = _require_finite_nonnegative(diameter, :diameter)
    return max(2 * gap0, Lf * D^2) / sqrt(k + 1)
end

function frank_wolfe_nonconvex_constant_step(; k, kwargs...)
    _require_integer_at_least(k, 0)
    return 1 / sqrt(k + 1)
end

function frank_wolfe_convex_curvature_bound(; k, curvature, kwargs...)
    _require_integer_at_least(k, 1)
    C = _require_finite_nonnegative(curvature, :curvature)
    return 2 * C / (k + 2)
end

function frank_wolfe_nonconvex_curvature_gap_bound(; k, initial, curvature, kwargs...)
    _require_integer_at_least(k, 0)
    gap0 = _require_finite_nonnegative(initial, :initial)
    C = _require_finite_nonnegative(curvature, :curvature)
    return max(2 * gap0, C) / sqrt(k + 1)
end
