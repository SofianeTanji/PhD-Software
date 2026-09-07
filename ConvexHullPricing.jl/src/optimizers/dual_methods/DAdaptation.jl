# ─────────────────────────────────────────────────────────────────────────────
# D-Adaptation Method  (see Defazio et al., 2023)
# ─────────────────────────────────────────────────────────────────────────────
# ── Core D-Adaptation ────────────────────────────────────────────────────────
# Unifies: DAdaptation, tDAdaptation, tsmoothDAdaptation
#
# When `smoothing_parameter` is `nothing` the exact oracle is used;
# otherwise the smooth oracle is called.
function DAdaptation(
    instance,
    initial_prices,
    stop::StoppingCriterion,
    initial_distance_estimate = PC;
    smoothing_parameter = nothing,
    verbose::Int = -1,
)
    oracle = build_oracle_subproblems(instance)
    T = length(instance.Load)
    iterates = [initial_prices]
    fun_iterates = Float64[]
    oracle_times = Float64[]
    grad_iterates = Vector{Float64}[]

    # Evaluate once to initialise Gamma
    _ot_init = @elapsed begin
        fun_oracle, grad_oracle = exact_oracle(oracle, iterates[1])
    end
    push!(oracle_times, _ot_init)
    S = [zeros(T)]
    D = [initial_distance_estimate]
    Gamma = [1 / norm(grad_oracle)]
    time_vector = [0.0]
    k = 1

    while should_continue(stop, k, time_vector[end])
        verbose > 0 && @info "[DAdaptation: Iteration $k; D = $(D[end])]"
        it_time = @elapsed begin
            # Oracle call — exact or smooth
            _ot = @elapsed begin
                if isnothing(smoothing_parameter)
                    fun_oracle, grad_oracle = exact_oracle(oracle, iterates[k])
                else
                    fun_oracle, grad_oracle = exact_smooth_oracle(
                        instance,
                        iterates[k],
                        smoothing_parameter,
                    )
                end
            end
            push!(oracle_times, _ot)
            push!(fun_iterates, fun_oracle)
            fun_oracle, grad_oracle = negate_for_minimization(fun_oracle, grad_oracle)  # max concave → min convex

            push!(grad_iterates, grad_oracle)
            push!(S, S[k] .+ D[k] .* grad_iterates[k])
            push!(Gamma, 1 / sqrt(sum(norm(grad_iterates[j])^2 for j = 1:k)))
            push!(
                D,
                max(
                    (
                        Gamma[k+1] * norm(S[k+1])^2 -
                        sum(Gamma[j] * D[j]^2 * norm(grad_iterates[j])^2 for j = 1:k)
                    ) / (2 * norm(S[k+1])),
                    D[k],
                ),
            )
            push!(iterates, ProjBox(iterates[1] - Gamma[k+1] * S[k+1], PCD, PCU))
        end
        push!(time_vector, it_time + time_vector[end])
        k += 1
        _lstar_reached(stop, fun_iterates[end]) && break
    end

    # The D-Adaptation convergence guarantee is on the weighted average,
    # but for consistency with all other optimizers, return the best iterate.
    x_best = iterates[argmax(fun_iterates)]
    return x_best, iterates, fun_iterates, time_vector, oracle_times
end
