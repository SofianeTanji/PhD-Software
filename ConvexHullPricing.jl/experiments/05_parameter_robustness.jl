# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 5: Parameter robustness (R8)
# ═══════════════════════════════════════════════════════════════════════════════
#
# One representative method per family (EST-POL, D-Adapt, FGM, PC-BPLM),
# 3 parameter values centered on the tuned value (±1 order of magnitude for
# log-scale methods; linear ±0.4 for level-type), 3 randomly selected
# instances per dataset (seed=42), 5-min budget.

using Random

function run_parameter_robustness()
    BE, CA = load_instances()
    dir = joinpath(RESULTS_DIR, "robustness")
    mkpath(dir)

    # Specs: (name, factory, level_type)
    # level_type=true  → α ∈ (0,1), use linear ±0.4 grid around tuned value
    # level_type=false → log-scale, use ×10^[-1,-0.5,0,0.5,1] around tuned value
    robustness_specs = [
        ("EST-POL", (i, x, α, b) -> EstimatedPolyak(i, x, TimeBudget(b), α), false),
        ("D-Adapt", (i, x, α, b) -> DAdaptation(i, x, TimeBudget(b), α), false),
        (
            "FGM",
            (i, x, α, b) -> FastGradientMethod(
                i,
                x,
                TimeBudget(b),
                1e-8;
                stepsize = α,
                use_shift = true,
                recompute_exact = true,
            ),
            false,
        ),
        (
            "PC-BPLM",
            (i, x, α, b) -> PreconditionedProximalLevelMethod(i, x, TimeBudget(b), α),
            true,
        ),
    ]

    results = DataFrame(
        method = String[],
        dataset = String[],
        tag = String[],
        param = Float64[],
        best_obj = Float64[],
    )

    # 3 randomly selected instances per dataset (fixed seed for reproducibility)
    rng = MersenneTwister(42)
    ds_insts = Dict(
        "BE" =>
            [(BE[i], "BE", be_tag(i)) for i in sort(shuffle(rng, 1:length(BE))[1:3])],
        "CA" =>
            [(CA[i], "CA", ca_tag(i)) for i in sort(shuffle(rng, 1:length(CA))[1:3])],
    )

    defaults = Dict(
        "EST-POL" => (ds -> _ds_param(ds, 32000.0, 300.0)),
        "D-Adapt" => (ds -> _ds_param(ds, 10.0, 0.15)),
        "FGM" => (ds -> _ds_param(ds, 1e-4, 1e-5)),
        "PC-BPLM" => (ds -> 0.3),
    )

    for (mname, factory, level_type) in robustness_specs
        for ds in ["BE", "CA"]
            α_tuned = _tuned_param(mname, ds, defaults[mname](ds))
            grid = if level_type
                clamp.(α_tuned .+ [-0.4, 0.0, 0.4], 0.01, 0.99)
            else
                α_tuned .* (10 .^ range(-1, 1, length = 3))
            end
            for (inst, _, tag) in ds_insts[ds]
                x0 = get_x0(inst)
                for α in grid
                    @info "Robustness: $mname α=$α on $tag"
                    try
                        result = factory(inst, x0, α, BUDGET_5)
                        fvals = result[3]
                        push!(results, (mname, ds, tag, α, maximum(fvals)))
                        jldsave(joinpath(dir, "robustness.jld2"); results)
                    catch e
                        @warn "  $mname α=$α on $tag failed" exception=e
                    end
                end
            end
        end
    end

    @info "Parameter robustness: $(nrow(results)) runs."
    return results
end
