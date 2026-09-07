# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 7: Main 15-minute benchmark — all methods, all instances
# ═══════════════════════════════════════════════════════════════════════════════
#
# Results saved to results/benchmark/{METHOD}_{TAG}.jld2
# Existing files are skipped (set overwrite=true to re-run).

function run_main_benchmark(; methods = METHOD_NAMES, budget = BUDGET, overwrite = false)
    BE, CA = load_instances()
    dir = joinpath(RESULTS_DIR, "benchmark")
    mkpath(dir)

    excluded_tags = Set(["CA9", "CA10", "CA11", "CA12"])

    for m in methods
        for (inst, ds, tag) in all_tagged_instances(BE, CA)
            if tag in excluded_tags
                @info "Skipping $tag (excluded)"
                continue
            end
            name = "$(m)_$(tag)"
            outfile = joinpath(dir, "$name.jld2")

            if !overwrite && isfile(outfile)
                @info "Skipping $name (exists)"
                continue
            end

            lstar = _get_ground_truth_value(tag)
            @info "Running $m on $tag" *
                  (isnothing(lstar) ? "" : " (L*=$(round(lstar; sigdigits=8)))")
            x0 = get_x0(inst)
            try
                if m == "Polyak"
                    if isnothing(lstar)
                        @warn "  No L* for $tag — skipping Polyak"
                        continue
                    end
                    result = run_method_polyak(inst, x0, budget, lstar)
                else
                    result = run_method(m, inst, x0, budget, ds; lstar = lstar)
                end
                save_run(dir, name, result)
                @info "  best = $(round(maximum(result[3]); sigdigits=8)), " *
                      "time = $(round(result[4][end]; digits=1))s"
            catch e
                @warn "  $m on $tag failed" exception=(e, catch_backtrace())
            end
        end
    end
    @info "Benchmark complete."
end


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 5: Primal DW/CG baseline
# ═══════════════════════════════════════════════════════════════════════════════

function run_primal_baseline()
    run_main_benchmark(; methods = ["CG"])
end
