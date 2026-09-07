# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 9: Preconditioner comparison for PC-BPLM
# ═══════════════════════════════════════════════════════════════════════════════
#
# Compares 5 preconditioner configurations against unpreconditioned MC-BPLM
# on 6 representative instances (3 Belgian, 3 Californian).

include("common.jl")

function run_preconditioner_comparison()
    BE, CA = load_instances()

    # ── Instance selection ────────────────────────────────────────────────────
    # BE1 (autumn), BE3 (spring), BE5 (summer); CA1, CA5, CA13
    instances = [
        (BE[1], "BE", "BE1"),
        (BE[3], "BE", "BE3"),
        (BE[5], "BE", "BE5"),
        (CA[1], "CA", "CA1"),
        (CA[5], "CA", "CA5"),
        (CA[13], "CA", "CA13"),
    ]

    dir = joinpath(RESULTS_DIR, "precond_comparison")
    mkpath(dir)

    budget = 90.0
    α = 0.9

    # Load ground truth for excess side payment
    Lstar = load_ground_truth()

    results = DataFrame(
        method = String[],
        tag = String[],
        best_obj = Float64[],
        excess_sp = Float64[],
        final_gap = Float64[],
        n_iters = Int[],
    )

    for (inst, ds, tag) in instances
        x0 = get_x0(inst)
        ub = fast_uc_upper_bound(inst; time_limit = 10.0)
        lstar_val = get(Lstar, tag, nothing)
        stop =
            isnothing(lstar_val) ? TimeBudget(budget) :
            TimeBudgetWithLstar(budget, lstar_val)

        # Static diagonal norm: w[t] = max(Load[t], P_max_tot - Load[t]) / ΔΠ
        T_len = length(inst.Load)
        P_max_tot = sum(inst.ThermalGen.MaxRunCapacity)
        ΔΠ = ConvexHullPricing.BLM_UPPER - ConvexHullPricing.BLM_LOWER
        static_w = [max(inst.Load[t], P_max_tot - inst.Load[t]) / ΔΠ for t = 1:T_len]

        methods = [
            ("MC-BPLM", nothing),
            ("PC-clipped", ClippedInversePrecond(p = 2.0, w_min = 1e-4, w_max = 1e4)),
        ]

        for (label, precond) in methods
            outfile = joinpath(dir, "$(label)_$(tag).jld2")
            skip_if_exists(outfile) && continue
            @info "Precond comparison: $label on $tag"
            try
                if isnothing(precond)
                    # MC-BPLM baseline (unpreconditioned)
                    result = MulticutBundleProximalLevelMethod(
                        inst,
                        x0,
                        stop,
                        α;
                        initial_ub_lstar = ub,
                        return_bounds = true,
                    )
                else
                    # PC-BPLM with specified preconditioner
                    result = PreconditionedProximalLevelMethod(
                        inst,
                        x0,
                        stop,
                        α;
                        initial_ub_lstar = ub,
                        return_bounds = true,
                        preconditioner = precond,
                    )
                end

                save_run_with_bounds(dir, "$(label)_$(tag)", result)

                _, iterates, fun_iterates, tvec, _, UBs, LBs = result
                best_obj = maximum(fun_iterates)
                n_iters = length(fun_iterates)
                final_gap = (isempty(UBs) || isempty(LBs)) ? Inf : UBs[end] - LBs[end]
                esp = isnothing(lstar_val) ? NaN : excess_side_payment(best_obj, lstar_val)

                push!(results, (label, tag, best_obj, esp, final_gap, n_iters))
            catch e
                @warn "  $label on $tag failed" exception=(e, catch_backtrace())
            end
        end
    end

    # ── Summary: geometric mean of excess side payment per method ─────────
    summary = combine(
        groupby(results, :method),
        :excess_sp =>
            (v -> begin
                finite = filter(x -> isfinite(x) && x > 0, v)
                isempty(finite) ? NaN : exp(mean(log.(finite)))
            end) => :geomean_esp,
        :n_iters => mean => :mean_iters,
        :final_gap => (v -> begin
            finite = filter(isfinite, v)
            isempty(finite) ? NaN : mean(finite)
        end) => :mean_gap,
    )

    jldsave(joinpath(dir, "precond_comparison.jld2"); results, summary)

    @info "Preconditioner comparison complete: $(nrow(results)) runs."
    println("\n── Per-instance results ──")
    show(stdout, results; allrows = true, allcols = true)
    println("\n\n── Summary (geometric mean excess SP) ──")
    show(stdout, summary; allrows = true, allcols = true)
    println()

    return results, summary
end

run_preconditioner_comparison()
