# ─────────────────────────────────────────────────────────────────────────────
# Polyak Step-Size Methods  (see Nesterov, 2006)
# ─────────────────────────────────────────────────────────────────────────────
# ── Exact Polyak method (requires optimal value) ────────────────────────────
# Unifies: PolyakMethod, tPolyakMethod
function PolyakMethod(
    instance,
    initial_prices,
    stop::StoppingCriterion,
    ObjSol;
    verbose::Int = -1,
)
    oracle = build_oracle_subproblems(instance)
    iterates = [initial_prices]
    fun_iterates = Float64[]
    oracle_times = Float64[]
    time_vector = [0.0]
    i = 1

    while should_continue(stop, i, time_vector[end])
        verbose > 0 && @info "[Polyak: Iteration $i]"
        it_time = @elapsed begin
            _ot = @elapsed begin
                fun_oracle, grad_oracle = exact_oracle(oracle, iterates[i])
            end
            push!(oracle_times, _ot)
            push!(fun_iterates, fun_oracle)
            fun_oracle, grad_oracle = negate_for_minimization(fun_oracle, grad_oracle)

            if norm(grad_oracle)^2 <= 1e-6
                break
            end
            PolyakStepsize = abs(fun_oracle + ObjSol) / norm(grad_oracle)^2
            push!(
                iterates,
                ProjBox(iterates[i] - 2 * PolyakStepsize * grad_oracle, PCD, PCU),
            )
        end
        push!(time_vector, it_time + time_vector[end])
        i += 1
    end
    _ot_final = @elapsed begin
        _final_val = exact_oracle(oracle, last(iterates))[1]
    end
    push!(oracle_times, _ot_final)
    push!(fun_iterates, _final_val)
    return last(iterates), iterates, fun_iterates, time_vector, oracle_times
end

# ── Estimated Polyak method (no optimal value needed) ───────────────────────
# Unifies: EstimatedPolyak, tEstimatedPolyak
function EstimatedPolyak(
    instance,
    initial_prices,
    stop::StoppingCriterion,
    alpha;
    verbose::Int = -1,
)
    oracle = build_oracle_subproblems(instance)
    iterates = [initial_prices]
    fun_iterates = Float64[]
    oracle_times = Float64[]
    time_vector = [0.0]
    i = 1

    while should_continue(stop, i, time_vector[end])
        verbose > 0 && @info "[Est. Polyak: Iteration $i]"
        it_time = @elapsed begin
            _ot = @elapsed begin
                fun_oracle, grad_oracle = exact_oracle(oracle, iterates[i])
            end
            push!(oracle_times, _ot)
            push!(fun_iterates, fun_oracle)
            fun_oracle, grad_oracle = negate_for_minimization(fun_oracle, grad_oracle)

            if norm(grad_oracle)^2 <= 1e-5
                break
            end
            PolyakStepsize =
                abs(fun_oracle + maximum(fun_iterates) + (alpha / i)) / norm(grad_oracle)^2
            push!(
                iterates,
                ProjBox(iterates[i] - 2 * PolyakStepsize * grad_oracle, PCD, PCU),
            )
        end
        push!(time_vector, it_time + time_vector[end])
        i += 1
        _lstar_reached(stop, fun_iterates[end]) && break
    end
    x_best = iterates[argmax(fun_iterates)]
    return x_best, iterates, fun_iterates, time_vector, oracle_times
end
