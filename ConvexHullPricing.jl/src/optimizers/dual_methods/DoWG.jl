# ─────────────────────────────────────────────────────────────────────────────
# DoWG Method  (see Khaled et al., 2023)
# ─────────────────────────────────────────────────────────────────────────────
# ── Core DoWG ────────────────────────────────────────────────────────────────
# Unifies: DowG, tDowG, tsmoothDowG
function DowG(
    instance,
    initial_prices,
    stop::StoppingCriterion,
    initial_distance_estimate = PC;
    smoothing_parameter = nothing,
    verbose::Int = -1,
)
    oracle = build_oracle_subproblems(instance)
    list_V = [0.0]
    list_R = [initial_distance_estimate]
    iterates = [initial_prices]
    fun_iterates = Float64[]
    oracle_times = Float64[]
    time_vector = [0.0]
    t = 1

    while should_continue(stop, t, time_vector[end])
        verbose > 0 && @info "[DowG: Iteration $t]"
        it_time = @elapsed begin
            push!(list_R, max(norm(iterates[t] - iterates[1]), list_R[t]))

            _ot = @elapsed begin
                if isnothing(smoothing_parameter)
                    fun_oracle, grad_oracle = exact_oracle(oracle, iterates[t])
                else
                    fun_oracle, grad_oracle = exact_smooth_oracle(
                        instance,
                        iterates[t],
                        smoothing_parameter,
                    )
                end
            end
            push!(oracle_times, _ot)
            push!(fun_iterates, fun_oracle)
            fun_oracle, grad_oracle = negate_for_minimization(fun_oracle, grad_oracle)

            push!(list_V, list_V[t] + list_R[t+1]^2 * norm(grad_oracle)^2)
            stepsize = list_R[t+1]^2 / sqrt(list_V[t+1])
            push!(iterates, ProjBox(iterates[t] - stepsize * grad_oracle, PCD, PCU))
        end
        push!(time_vector, it_time + time_vector[end])
        t += 1
        _lstar_reached(stop, fun_iterates[end]) && break
    end
    x_best = iterates[argmax(fun_iterates)]
    return x_best, iterates, fun_iterates, time_vector, oracle_times
end
