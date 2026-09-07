# Experiments

Run scripts from their experiment directory with the repository environment.
The scripts and notebooks use `experiments/solver_utils.py` to compute BADR
weights with the toolbox's SLSQP solver. The helper optimizes the training metric,
checks convergence and simplex feasibility, and raises an error on failure.
There is no grid or random-search fallback.

These experiment calls use SLSQP's three-point numerical derivatives of the
fitted-model objective. This includes the effects of model preprocessing and
intercepts without relying on the implicit oracle's analytical derivative.
The SLSQP class still uses the oracle gradient by default for existing callers;
set `use_oracle_gradient=False` to request numerical derivatives explicitly.

The scalability BADR-SGD runs continue to execute BADR-SGD. Their reference
objective is now computed by SLSQP, rather than a grid minimum. It is a numerical
reference, not a certified global optimum.

Grids remain for drawing Pareto curves and heatmaps. BADR markers on those plots
come from the solver, independently of the plotting resolution. One-Fit retains
its definition of selecting among group-specific models; Frank-Wolfe retains
its simplex linear minimization step.

## New outputs

To avoid resuming earlier runs that used sampled minima, the revised scripts use:

- Experiment 2: `results_slsqp.jsonl` and `results_slsqp_completed_keys.json`.
  Its notebook and plotter read the new results file.
- Experiment 3: `outputs_slsqp/`.
- Scalability: `fat_experiment_slsqp_reference.json`.

Affected notebook outputs have been cleared. Existing result files and figures
are historical artifacts and must be regenerated before presenting them as
outputs of the revised solver-based experiments.
