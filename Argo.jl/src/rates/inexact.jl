function _accuracy_schedule_at(δ::Real, index::Integer)
    _require_integer_at_least(index, 1, :index)
    return _require_finite_nonnegative(δ, :proximal_accuracy)
end

function _accuracy_schedule_at(δ::Number, index::Integer)
    _require_integer_at_least(index, 1, :index)
    return _require_finite_nonnegative(δ, :proximal_accuracy)
end

function _accuracy_schedule_at(schedule::Function, index::Integer)
    _require_integer_at_least(index, 1, :index)
    return _require_finite_nonnegative(schedule(index), :proximal_accuracy)
end

function _accuracy_schedule_at(schedule, index::Integer)
    _require_integer_at_least(index, 1, :index)
    values = Tuple(schedule)
    length(values) >= index ||
        throw(ArgumentError("proximal accuracy schedule has fewer than k entries"))
    return _require_finite_nonnegative(values[index], :proximal_accuracy)
end

function _proximal_accuracy_schedule(proximal_accuracy, proximal_accuracy_schedule)
    proximal_accuracy_schedule === nothing || return proximal_accuracy_schedule
    proximal_accuracy === nothing &&
        throw(ArgumentError("proximal_accuracy or proximal_accuracy_schedule is required"))
    return proximal_accuracy
end

function schmidt2011_inexact_proximal_gradient_average_bound(;
    k,
    initial,
    L,
    step_size=nothing,
    proximal_accuracy=nothing,
    proximal_accuracy_schedule=nothing,
    kwargs...,
)
    _require_integer_at_least(k, 1)
    R2 = _require_finite_nonnegative(initial, :initial)
    γ = if step_size === nothing
        1 / _require_finite_positive(L, :L)
    else
        _require_finite_positive(step_size, :step_size)
    end
    schedule = _proximal_accuracy_schedule(proximal_accuracy, proximal_accuracy_schedule)
    sum_delta = 0.0
    sum_sqrt = 0.0
    for index in 1:Int(k)
        δ = _accuracy_schedule_at(schedule, index)
        sum_delta += δ
        sum_sqrt += sqrt(2 * γ * δ)
    end
    radius = sqrt(R2)
    return (radius + 2 * sum_sqrt + sqrt(2 * γ * sum_delta))^2 / (2 * γ * k)
end

"""
    schmidt2011_inexact_proximal_gradient_average_bound(; k, initial, L, proximal_accuracy)

Args: iteration count, initial distance, smoothness, and proximal accuracy schedule.
Returns: scalar bound.
- Evaluates the Schmidt 2011 inexact proximal-gradient average guarantee.
"""
schmidt2011_inexact_proximal_gradient_average_bound
