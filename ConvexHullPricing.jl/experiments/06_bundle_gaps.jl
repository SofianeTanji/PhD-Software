# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 6: Bundle gap analysis (R9)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Run the retained bundle methods with return_bounds=true until the bundle gap
# reaches the manuscript target, with the standard 15-minute budget kept only as
# a safety cap.

function run_bundle_gap_analysis()
    BE, CA = load_instances()
    dir = joinpath(RESULTS_DIR, "bundle_gaps")
    mkpath(dir)
    methods = ["PC-BLM", "PC-BPLM"]
    excluded_tags = Set(["CA9", "CA10", "CA11", "CA12"])
    lstar = load_ground_truth()

    results = DataFrame(
        method = String[],
        dataset = String[],
        tag = String[],
        target_gap = Float64[],
        final_UB = Float64[],
        final_LB = Float64[],
        final_gap = Float64[],
        elapsed_time = Float64[],
        reached_target = Bool[],
        time_to_target = Float64[],
    )

    build_result_row(method, ds, tag, target_gap, path) = begin
        d = load(path)
        UBs = haskey(d, "UpperBounds") ? d["UpperBounds"] : Float64[]
        LBs = haskey(d, "LowerBounds") ? d["LowerBounds"] : Float64[]
        tvec = haskey(d, "time_vector") ? d["time_vector"] : Float64[]
        final_UB = isempty(UBs) ? Inf : UBs[end]
        final_LB = isempty(LBs) ? -Inf : LBs[end]
        final_gap = final_UB - final_LB
        elapsed_time = isempty(tvec) ? Inf : tvec[end]
        reached_target = isfinite(final_gap) && final_gap <= target_gap
        time_to_target = reached_target ? elapsed_time : Inf
        return (
            method,
            ds,
            tag,
            target_gap,
            final_UB,
            final_LB,
            final_gap,
            elapsed_time,
            reached_target,
            time_to_target,
        )
    end

    for m in methods
        for (inst, ds, tag) in all_tagged_instances(BE, CA)
            tag in excluded_tags && continue
            haskey(lstar, tag) || (@warn "No ground-truth value for $tag; skipping"; continue)
            @info "Bundle gaps: $m on $tag"
            x0 = get_x0(inst)
            target_gap = TIME_TO_THRESHOLD_FACTOR * lstar[tag]
            outfile = joinpath(dir, "$(m)_$(tag).jld2")
            if isfile(outfile)
                @info "Reusing existing: $outfile"
                push!(results, build_result_row(m, ds, tag, target_gap, outfile))
                continue
            end
            try
                stop = GapToleranceWithBudget(BUDGET, target_gap)
                result = run_method(
                    m,
                    inst,
                    x0,
                    BUDGET,
                    ds;
                    stop = stop,
                    return_bounds = true,
                )
                save_run_with_bounds(dir, "$(m)_$(tag)", result)
                push!(results, build_result_row(m, ds, tag, target_gap, outfile))
                jldsave(joinpath(dir, "gap_analysis.jld2"); results)
            catch e
                @warn "  $m on $tag failed" exception=(e, catch_backtrace())
            end
        end
    end

    summary = combine(
        groupby(results, :method),
        :reached_target => sum => :n_reached,
        :reached_target => length => :n_total,
        :time_to_target =>
            (t -> (v = filter(isfinite, t); isempty(v) ? NaN : mean(v))) =>
                :mean_time_reached,
        :time_to_target =>
            (t -> (v = filter(isfinite, t); isempty(v) ? NaN : maximum(v))) =>
                :max_time_reached,
    )

    dataset_summary = combine(
        groupby(results, [:method, :dataset]),
        :reached_target => sum => :n_reached,
        :reached_target => length => :n_total,
        :time_to_target =>
            (t -> (v = filter(isfinite, t); isempty(v) ? NaN : mean(v))) =>
                :mean_time_reached,
        :time_to_target =>
            (t -> (v = filter(isfinite, t); isempty(v) ? NaN : maximum(v))) =>
                :max_time_reached,
    )

    jldsave(joinpath(dir, "gap_analysis.jld2"); results, summary, dataset_summary)
    @info "Bundle gap analysis: $(nrow(results)) runs.\n$summary"
    return results, summary, dataset_summary
end
