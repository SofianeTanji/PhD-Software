abstract type StoppingCriterion end

"""
    IterationLimit(n::Int)

Stop after exactly `n` iterations.
"""
struct IterationLimit <: StoppingCriterion
    max_iterations::Int
end

"""
    TimeBudget(seconds::Float64)

Stop when cumulative wall-clock time exceeds `seconds`.
"""
struct TimeBudget <: StoppingCriterion
    max_seconds::Float64
end

"""
    GapTolerance(tol::Float64)

Stop when `upper_bound - lower_bound < tol`.  Used by bundle-level methods.
"""
struct GapTolerance <: StoppingCriterion
    tolerance::Float64
end

"""
    GapToleranceWithBudget(max_seconds::Float64, tol::Float64)

Stop when `upper_bound - lower_bound < tol`, with `max_seconds` as a safety cap.
Used by bundle methods when the experiment should terminate on a target gap but
still guard against pathological runs.
"""
struct GapToleranceWithBudget <: StoppingCriterion
    max_seconds::Float64
    tolerance::Float64
end

"""
    TimeBudgetWithLstar(max_seconds::Float64, Lstar::Float64)

Stop when cumulative wall-clock time exceeds `max_seconds` OR the best
dual function value reaches `Lstar`.
"""
struct TimeBudgetWithLstar <: StoppingCriterion
    max_seconds::Float64
    Lstar::Float64
end

"""Return `true` if the loop should execute iteration `iter`."""
should_continue(c::IterationLimit, iter::Int, elapsed::Float64) = iter <= c.max_iterations
should_continue(c::TimeBudget, iter::Int, elapsed::Float64) = elapsed <= c.max_seconds
should_continue(c::TimeBudgetWithLstar, iter::Int, elapsed::Float64) =
    elapsed <= c.max_seconds
should_continue(c::GapTolerance, iter::Int, gap::Float64) = gap > c.tolerance
should_continue(c::GapToleranceWithBudget, iter::Int, elapsed::Float64) =
    elapsed <= c.max_seconds

"""
Return `true` if the most recent function value `f` has reached or exceeded `Lstar`.
Default (for all other stopping criteria): always returns `false`.
"""
_lstar_reached(::StoppingCriterion, ::Float64) = false
_lstar_reached(c::TimeBudgetWithLstar, f::Float64) = f >= c.Lstar

"""For criteria that need the total iteration count known up-front."""
max_iterations(c::IterationLimit) = c.max_iterations

"""
    negate_for_minimization(fun, grad) -> (-fun, -grad)

The Lagrangian oracles return `(value, gradient)` of the *concave* dual function
(to be maximised).  Dual optimizers work by minimising the negative, so this
helper flips both signs.

**Convention:** record the *original* `fun` in `fun_iterates` (it represents the
Lagrangian dual bound) *before* calling this function.
"""
negate_for_minimization(fun, grad) = (-fun, -grad)
