# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 1: Ground-truth L* computation
# ═══════════════════════════════════════════════════════════════════════════════
#
# PC-PBLM (α=0.4) with exact oracle (MIP gap = 0) until UB−LB < $1.
# Takes up to ~6 hours per instance.

function run_ground_truth()
    BE, CA = load_instances()
    dir = joinpath(RESULTS_DIR, "ground_truth")

    mkpath(dir)
    for (inst, ds, tag) in all_tagged_instances(BE, CA)
        outfile = joinpath(dir, "GT_$tag.jld2")
        skip_if_exists(outfile) && continue
        @info "Ground truth $tag"
        try
            x0 = get_x0(inst)
            result = PreconditionedProximalLevelMethod(
                inst,
                x0,
                GapTolerance(1.0),
                0.3;
                verbose = 1,
            )
            save_run(dir, "GT_$tag", result)
            @info "  L* = $(isempty(result[3]) ? "(no iterates)" : maximum(result[3])), time = $(round(result[4][end]; digits=0))s"
        catch e
            @warn "Ground truth $tag failed" exception=(e, catch_backtrace())
        end
    end
end
