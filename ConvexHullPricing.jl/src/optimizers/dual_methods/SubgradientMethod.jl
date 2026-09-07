# ─────────────────────────────────────────────────────────────────────────────
# Subgradient Methods  (see Nesterov, 2006)
# ─────────────────────────────────────────────────────────────────────────────
# ── Core subgradient method ─────────────────────────────────────────────────
function SubgradientMethod(
    instance,
    initial_prices,
    stop::StoppingCriterion,
    alpha;
    normalize_gradient::Bool = true,
    verbose::Int = -1,
)
    oracle = build_oracle_subproblems(instance)
    iterates = [initial_prices]
    fun_iterates = Float64[]
    oracle_times = Float64[]
    time_vector = [0.0]
    i = 1

    while should_continue(stop, i, time_vector[end])
        verbose > 0 && @info "[SubG: Iteration $i]"
        it_time = @elapsed begin
            _ot = @elapsed begin
                fun_oracle, grad_oracle = exact_oracle(oracle, iterates[i])
            end
            push!(oracle_times, _ot)
            push!(fun_iterates, fun_oracle)
            fun_oracle, grad_oracle = negate_for_minimization(fun_oracle, grad_oracle)

            denom = normalize_gradient ? (norm(grad_oracle) * i) : i
            push!(iterates, ProjBox(iterates[i] - (alpha / denom) * grad_oracle, PCD, PCU))
        end
        push!(time_vector, it_time + time_vector[end])
        i += 1
        _lstar_reached(stop, fun_iterates[end]) && break
    end
    x_best = iterates[argmax(fun_iterates)]
    return x_best, iterates, fun_iterates, time_vector, oracle_times
end

# ── Last-iterate optimal subgradient method ─────────────────────────────────
# Unifies: lastSubgradientMethod, tlastSubgradientMethod
# Note: stepsize depends on total iteration count, so `niter` is required.
function LastIterateSubgradientMethod(
    instance,
    initial_prices,
    niter::Int,
    R;
    Lstar::Float64 = Inf,
    verbose::Int = -1,
)
    oracle = build_oracle_subproblems(instance)
    iterates = [initial_prices]
    fun_iterates = Float64[]
    oracle_times = Float64[]
    time_vector = [0.0]

    for k = 1:niter
        verbose > 0 && @info "[LastSubG: Iteration $k]"
        it_time = @elapsed begin
            _ot = @elapsed begin
                fun_oracle, grad_oracle = exact_oracle(oracle, iterates[k])
            end
            push!(oracle_times, _ot)
            push!(fun_iterates, fun_oracle)
            fun_oracle, grad_oracle = negate_for_minimization(fun_oracle, grad_oracle)

            stepsize = R * (niter + 1 - k) / sqrt((niter + 1)^3)
            push!(
                iterates,
                ProjBox(iterates[k] - stepsize * grad_oracle / norm(grad_oracle), PCD, PCU),
            )
        end
        push!(time_vector, it_time + time_vector[end])
        fun_iterates[end] >= Lstar && break
    end
    x_best = iterates[argmax(fun_iterates)]
    return x_best, iterates, fun_iterates, time_vector, oracle_times
end

