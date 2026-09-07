# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 4: Hyperparameter tuning — coarse + refined grid search
# ═══════════════════════════════════════════════════════════════════════════════
#
# For each method's single hyperparameter: 6-value coarse grid (IterationLimit),
# then 3-value refined grid around the best (TimeBudget).
# 10 methods are tuned directly; CG and CRG use fixed values; Polyak requires
# the exact optimum (no free hyperparameter).
# Tuning instance = first instance of each dataset.

function run_hyperparameter_tuning()
    BE, CA = load_instances()
    dir = joinpath(RESULTS_DIR, "tuning")
    mkpath(dir)

    # Each spec: (name, hp_name, factory(inst, x0, α, stop) → 4-tuple, grid_BE, grid_CA)
    specs = [
        (
            "SUBG-L",
            "R",
            (i, x, α, stop) -> begin
                if stop isa IterationLimit
                    return LastIterateSubgradientMethod(i, x, stop.max_iterations, α)
                else
                    _, _, _, tvec_pilot = LastIterateSubgradientMethod(i, x, 20, α)
                    t_per_iter = tvec_pilot[end] / 20
                    N = max(10, round(Int, stop.max_seconds / t_per_iter))
                    return LastIterateSubgradientMethod(i, x, N, α)
                end
            end,
            10 .^ range(-1, 3, length = 6),
            10 .^ range(-2, 1, length = 6),
        ),
        (
            "EST-POL",
            "α",
            (i, x, α, stop) -> EstimatedPolyak(i, x, stop, α),
            10 .^ range(2, 5, length = 6),
            10 .^ range(1, 4, length = 6),
        ),
        (
            "D-Adapt",
            "α",
            (i, x, α, stop) -> DAdaptation(i, x, stop, α),
            10 .^ range(-1, 2, length = 6),
            10 .^ range(-2, 1, length = 6),
        ),
        (
            "DoWG",
            "α",
            (i, x, α, stop) -> DowG(i, x, stop, α),
            10 .^ range(-1, 2, length = 6),
            10 .^ range(-2, 1, length = 6),
        ),
        (
            "FGM",
            "stepsize",
            (i, x, α, stop) -> FastGradientMethod(
                i,
                x,
                stop,
                1e-8;
                stepsize = α,
                use_shift = true,
                recompute_exact = true,
            ),
            10 .^ range(-6, -2, length = 6),
            10 .^ range(-7, -3, length = 6),
        ),
        (
            "SUBG",
            "α",
            (i, x, α, stop) ->
                SubgradientMethod(i, x, stop, α; normalize_gradient = true),
            10 .^ range(-1, 3, length = 6),
            10 .^ range(-2, 1, length = 6),
        ),
        (
            "PC-BLM",
            "α",
            (i, x, α, stop) -> PreconditionedLevelMethod(i, x, stop, α),
            range(0.1, 0.99, length = 6),
            range(0.1, 0.99, length = 6),
        ),
        (
            "PC-BPLM",
            "α",
            (i, x, α, stop) -> PreconditionedProximalLevelMethod(i, x, stop, α),
            range(0.1, 0.99, length = 6),
            range(0.1, 0.99, length = 6),
        ),
        (
            "DLM",
            "α",
            (i, x, α, stop) -> DynamicLevelMethod(i, x, stop, α),
            range(0.1, 0.99, length = 6),
            range(0.1, 0.99, length = 6),
        ),
    ]

    all_results = DataFrame(
        method = String[],
        dataset = String[],
        phase = String[],
        param = Float64[],
        best_obj = Float64[],
    )

    ds_first = Dict("BE" => BE[1], "CA" => CA[1])

    for ds in ["BE", "CA"]
        inst = ds_first[ds]
        x0 = get_x0(inst)

        for (mname, hp_name, factory, grid_BE, grid_CA) in specs
            grid = ds == "BE" ? grid_BE : grid_CA

            # Phase 1: coarse — fast screening with IterationLimit(10)
            @info "Tuning $mname on $ds — coarse ($(length(grid)) values)"
            best_val, best_α = -Inf, first(grid)
            for α in grid
                try
                    _, _, fvals, _ = factory(inst, x0, α, IterationLimit(10))
                    obj = maximum(fvals)
                    push!(all_results, (mname, ds, "coarse", α, obj))
                    if obj > best_val
                        best_val, best_α = obj, α
                    end
                catch e
                    @warn "  $mname α=$α failed" exception=e
                end
            end

            # Phase 2: refined — 3 values around coarse best with TimeBudget
            if best_α > 0 && best_α < 1   # level-type ∈ (0,1)
                lo = max(0.01, best_α - 0.1)
                hi = min(0.99, best_α + 0.1)
                refined = collect(range(lo, hi, length = 3))
            else                            # log-scale
                lp = log10(abs(best_α) + 1e-20)
                refined = collect(10 .^ range(lp - 0.3, lp + 0.3, length = 3))
            end

            @info "  Refined around $(round(best_α; sigdigits=3))"
            for α in refined
                try
                    _, _, fvals, _ = factory(inst, x0, α, TimeBudget(BUDGET_TUNE))
                    obj = maximum(fvals)
                    push!(all_results, (mname, ds, "refined", α, obj))
                    if obj > best_val
                        best_val, best_α = obj, α
                    end
                catch e
                    @warn "  $mname α=$α failed" exception=e
                end
            end
            @info "  → best α=$(round(best_α; sigdigits=4)), obj=$(round(best_val; sigdigits=8))"
            jldsave(joinpath(dir, "tuning_results.jld2"); all_results)
        end

        # ── Inject shared / fixed results for methods not tuned directly ──

        # Fixed values for CG and CRG
        push!(all_results, ("CG", ds, "fixed", 1e-6, NaN))
        push!(all_results, ("CRG", ds, "fixed", 1e-6, NaN))
    end

    # ── Markdown summary table ──
    function _grid_desc(grid)
        g = collect(grid)
        lo, hi = minimum(g), maximum(g)
        if hi / lo > 20
            "10^[$(round(log10(lo),digits=1)), $(round(log10(hi),digits=1))] (6 pts)"
        else
            "[$(round(lo,sigdigits=2)), $(round(hi,sigdigits=2))] (6 pts)"
        end
    end
    function _refined_desc(grid)
        g = collect(grid)
        all(0 .< g .< 1) ? "linear ±0.1 (3 pts)" : "log ×10^±0.3 (3 pts)"
    end
    function _best_param(method, ds, phase)
        rows = filter(
            r -> r.method == method && r.dataset == ds && r.phase == phase,
            all_results,
        )
        nrow(rows) == 0 && return "—"
        string(round(rows.param[argmax(rows.best_obj)], sigdigits = 4))
    end

    header = "| Method | Hyperparameter | Coarse grid (BE) | Coarse grid (CA) | Refined grid | Best coarse (BE) | Best coarse (CA) | Best refined (BE) | Best refined (CA) |"
    sep = "|--------|---------------|-----------------|-----------------|-------------|-----------------|-----------------|------------------|------------------|"
    md_rows = [header, sep]
    for (mname, hp_name, _, grid_BE, grid_CA) in specs
        push!(
            md_rows,
            "| $mname | $hp_name | $(_grid_desc(grid_BE)) | $(_grid_desc(grid_CA)) | $(_refined_desc(grid_BE)) | $(_best_param(mname,"BE","coarse")) | $(_best_param(mname,"CA","coarse")) | $(_best_param(mname,"BE","refined")) | $(_best_param(mname,"CA","refined")) |",
        )
    end
    md_table = join(md_rows, "\n")
    write(joinpath(dir, "tuning_summary.md"), md_table * "\n")
    @info "Tuning summary:\n$md_table"

    @info "Tuning complete."
    return all_results
end


# ═══════════════════════════════════════════════════════════════════════════════
# Hyperparameter tuning for SUBG only
# ═══════════════════════════════════════════════════════════════════════════════
#
# Re-tune SUBG after behavior changes. Updates existing tuning_results.jld2.

function retune_subg_only()
    BE, CA = load_instances()
    dir = joinpath(RESULTS_DIR, "tuning")
    mkpath(dir)

    # Load existing results or start fresh
    path = joinpath(dir, "tuning_results.jld2")
    if isfile(path)
        all_results = load(path)["all_results"]
        # Remove old SUBG rows
        all_results = filter(r -> r.method != "SUBG", all_results)
    else
        all_results = DataFrame(
            method = String[],
            dataset = String[],
            phase = String[],
            param = Float64[],
            best_obj = Float64[],
        )
    end

    ds_first = Dict("BE" => BE[1], "CA" => CA[1])

    for ds in ["BE", "CA"]
        inst = ds_first[ds]
        x0 = get_x0(inst)
        grid_BE = 10 .^ range(-1, 3, length = 6)
        grid_CA = 10 .^ range(-2, 1, length = 6)
        grid = ds == "BE" ? grid_BE : grid_CA

        mname = "SUBG"
        factory =
            (i, x, α, stop) -> SubgradientMethod(i, x, stop, α; normalize_gradient = true)

        @info "Tuning $mname on $ds — coarse ($(length(grid)) values)"
        best_val, best_α = -Inf, first(grid)
        for α in grid
            try
                _, _, fvals, _ = factory(inst, x0, α, IterationLimit(10))
                obj = maximum(fvals)
                push!(all_results, (mname, ds, "coarse", α, obj))
                if obj > best_val
                    best_val, best_α = obj, α
                end
            catch e
                @warn "  $mname α=$α failed" exception=e
            end
        end

        # Phase 2: refined — 3 values around coarse best with TimeBudget
        if best_α > 0 && best_α < 1   # level-type ∈ (0,1)
            lo = max(0.01, best_α - 0.1)
            hi = min(0.99, best_α + 0.1)
            refined = collect(range(lo, hi, length = 3))
        else                            # log-scale
            lp = log10(abs(best_α) + 1e-20)
            refined = collect(10 .^ range(lp - 0.3, lp + 0.3, length = 3))
        end

        @info "  Refined around $(round(best_α; sigdigits=3))"
        for α in refined
            try
                _, _, fvals, _ = factory(inst, x0, α, TimeBudget(BUDGET_TUNE))
                obj = maximum(fvals)
                push!(all_results, (mname, ds, "refined", α, obj))
                if obj > best_val
                    best_val, best_α = obj, α
                end
            catch e
                @warn "  $mname α=$α failed" exception=e
            end
        end
        @info "  → best α=$(round(best_α; sigdigits=4)), obj=$(round(best_val; sigdigits=8))"
    end

    jldsave(path; all_results)
    @info "SUBG tuning complete. Results saved to $path"
    return all_results
end


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 1b: SUBG-L iteration-count calibration
# ═══════════════════════════════════════════════════════════════════════════════
#
# SUBG-L uses a fixed iteration count N instead of a time budget.
# We calibrate N so that the total runtime ≈ 15 minutes.

function calibrate_subgl()
    BE, CA = load_instances()
    dir = joinpath(RESULTS_DIR, "tuning")
    mkpath(dir)

    results = Dict{String,Any}()

    for (ds, inst) in [("BE", BE[1]), ("CA", CA[1])]
        R = _tuned_param("SUBG-L", ds, _ds_param(ds, 40.0, 0.2))
        x0 = get_x0(inst)
        @info "Calibrating SUBG-L on $ds (R=$R)"

        # Pilot: run 20 iterations to estimate per-iteration cost
        _, _, _, tvec_pilot = LastIterateSubgradientMethod(inst, x0, 20, R)
        t_per_iter = tvec_pilot[end] / 20
        N_est = round(Int, BUDGET / t_per_iter)
        @info "  Pilot: $(round(t_per_iter; digits=3))s/iter → N_est=$N_est"

        # Search: try 5 candidates around the estimate
        candidates = unique(
            sort([
                max(10, N_est - 200),
                max(10, N_est - 50),
                N_est,
                N_est + 50,
                N_est + 200,
            ]),
        )
        N_best, t_best = N_est, Inf
        for N in candidates
            _, _, _, tvec = LastIterateSubgradientMethod(inst, x0, N, R)
            t = tvec[end]
            @info "    N=$N → $(round(t; digits=1))s"
            if abs(t - BUDGET) < abs(t_best - BUDGET)
                N_best, t_best = N, t
            end
        end

        results["N_$ds"] = N_best
        results["R_$ds"] = R
        results["runtime_$ds"] = t_best
        @info "  → N=$N_best ($(round(t_best; digits=1))s)"
    end

    jldsave(
        joinpath(dir, "subgl_calibration.jld2");
        (Symbol(k) => v for (k, v) in results)...,
    )
    return results
end
