# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 3: FGM smoothing sensitivity (Table 3)
# ═══════════════════════════════════════════════════════════════════════════════
#
# FGM for 30s across ς ∈ {1e-9,1e-8,1e-7} × η ∈ {1e-5,1e-4,1e-3},
# on all instances per dataset to confirm negligible sensitivity to ς
# across both datasets. Total budget: 28 instances × 9 combos × 30s = 126 min.

include("common.jl")
using Plots

const BUDGET_SENS = 30.0  # 30 seconds per run

function run_fgm_smoothing_sensitivity()
    BE, CA = load_instances()
    dir = joinpath(RESULTS_DIR, "fgm_smoothing")
    mkpath(dir)

    smoothings = [1e-9, 1e-8, 1e-7]
    steps = [1e-5, 1e-4, 1e-3]

    tag_fn = Dict("BE" => be_tag, "CA" => ca_tag)

    results = DataFrame(
        dataset = String[],
        tag = String[],
        idx = Int[],
        smoothing = Float64[],
        step = Float64[],
        best_obj = Float64[],
        n_iters = Int[],
    )

    for (ds, data) in [("BE", BE), ("CA", CA)]
        for (idx, inst) in enumerate(data)
            tag = tag_fn[ds](idx)
            x0 = get_x0(inst)
            for ς in smoothings, η in steps
                @info "FGM sens: $tag ς=$ς η=$η"
                result = FastGradientMethod(
                    inst,
                    x0,
                    TimeBudget(BUDGET_SENS),
                    ς;
                    stepsize = η,
                    use_shift = true,
                    recompute_exact = true,
                )
                fvals = result[3]
                push!(results, (ds, tag, idx, ς, η, maximum(fvals), length(fvals)))
                jldsave(joinpath(dir, "sensitivity.jld2"); results)
            end
        end
    end
    @info "FGM sensitivity saved."
    return results
end
