# ─────────────────────────────────────────────────────────────────────────────
# Fast / Accelerated Gradient Methods  (see Kim et al., 2016)
# ─────────────────────────────────────────────────────────────────────────────
# ─── Helper: pick the right oracle based on options ─────────────────────────
function _call_oracle(oracle, instance, x, μ, shift)
    if !isnothing(shift) && !isnothing(μ)
        return exact_translate_smooth_oracle(instance, x, μ, shift)
    elseif !isnothing(μ)
        return exact_smooth_oracle(instance, x, μ)
    else
        return exact_oracle(oracle, x)
    end
end

# ── Fast Gradient Method (Nesterov momentum) ────────────────────────────────
# Unifies: FastGradientMethod, tFastGradientMethod, tShiftedFGM
#
# - `smoothing_parameter` : μ for the smooth oracle  (required)
# - `use_shift`           : if true, computes LP relaxation shift and uses
#                           translate_smooth_oracle; also projects iterates
# - `stepsize`            : step size α  (1/Lips when not shifted)
# - `recompute_exact`     : re-evaluate true oracle after convergence
function FastGradientMethod(
    instance,
    X0,
    stop::StoppingCriterion,
    smoothing_parameter;
    stepsize = 1.0,
    use_shift::Bool = false,
    recompute_exact::Bool = false,
    verbose::Int = -1,
)
    oracle = build_oracle_subproblems(instance)
    shift = use_shift ? GetShift(instance) : nothing
    projected = use_shift  # shift ⟹ constrained ⟹ project

    x_iterates = [X0]
    y_iterates = [X0]
    fun_iterates = Float64[]
    oracle_times = Float64[]
    time_vector = [0.0]
    t = 1

    while should_continue(stop, t, time_vector[end])
        verbose > 0 && @info "[FGM: Iteration $t]"
        it_time = @elapsed begin
            _ot = @elapsed begin
                fun_oracle, grad_oracle = _call_oracle(
                    oracle,
                    instance,
                    y_iterates[t],
                    smoothing_parameter,
                    shift,
                )
            end
            push!(oracle_times, _ot)
            push!(fun_iterates, fun_oracle)
            fun_oracle, grad_oracle = negate_for_minimization(fun_oracle, grad_oracle)

            x_new = y_iterates[t] - stepsize * grad_oracle
            if projected
                x_new = ProjBox(x_new, PCD, PCU)
            end
            push!(x_iterates, x_new)

            y_new =
                x_iterates[t+1] + ((t - 1) / (t + 2)) * (x_iterates[t+1] - x_iterates[t])
            push!(y_iterates, y_new)
        end
        push!(time_vector, it_time + time_vector[end])
        t += 1
        _lstar_reached(stop, fun_iterates[end]) && break
    end

    # Optionally recompute true (non-smooth) objective values
    if recompute_exact
        verbose > 0 && @info "[FGM: Done. Re-computing true function values.]"
        fun_iterates = Float64[exact_oracle(oracle, ρ)[1] for ρ in x_iterates]
    end

    return last(x_iterates), x_iterates, fun_iterates, time_vector, oracle_times
end

