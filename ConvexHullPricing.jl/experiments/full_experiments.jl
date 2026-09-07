# ═══════════════════════════════════════════════════════════════════════════════
# full_experiments.jl — Orchestrator for the full experimental pipeline
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each experiment lives in its own file (01_*.jl through 08_*.jl).
# This file includes them all and provides `run_all()`.
#
# run_all() runs the core pipeline: ground truth → main benchmark → figures.
# Supporting experiments (02, 03, 04, 05, 06) must be run individually.
# Experiment 07 requires 01 (Polyak needs L*) and 04 (tuned params).
# Experiment 08 requires all.
# Note: compare_bundle.jl is a standalone script (not included here).
#
# Usage — interactive:
#   julia> include("full_experiments.jl")
#   julia> run_ground_truth()                # Experiment 1
#   julia> run_main_benchmark()              # Experiment 7
#   julia> generate_all_tables_and_figures() # Experiment 8
#
# Usage — full pipeline:
#   julia --threads=auto full_experiments.jl

using Revise
include("common.jl")                     # imports, constants, helpers, dispatch
include("01_ground_truth.jl")           # run_ground_truth()
include("02_milp_profiling.jl")         # run_milp_profiling()
include("03_fgm_sensitivity.jl")        # run_fgm_smoothing_sensitivity()
include("04_hyperparameter_tuning.jl")  # run_hyperparameter_tuning(), calibrate_subgl()
include("05_parameter_robustness.jl")   # run_parameter_robustness()
include("06_bundle_gaps.jl")            # run_bundle_gap_analysis()
include("07_main_benchmark.jl")         # run_main_benchmark(), run_primal_baseline()
include("08_tables_and_figures.jl")     # generate_all_tables_and_figures() + all generate_*()


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

function run_all()
    @info "═══ Full experimental pipeline ═══"
    t0 = time()

    @info "Step 1/3: Ground-truth L*"
    run_ground_truth()

    @info "Step 2/3: Main 15-minute benchmark (all methods)"
    run_main_benchmark()

    @info "Step 3/3: Tables and figures"
    generate_all_tables_and_figures()

    elapsed = round((time() - t0) / 3600; digits = 1)
    @info "═══ Pipeline complete in $(elapsed) hours ═══"
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all()
end
