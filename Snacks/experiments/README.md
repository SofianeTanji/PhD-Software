# Experiments

Research scripts for the Nyström SVM thesis experiments.

From the Snacks directory, install their dependencies with
`uv sync --no-dev --extra repro` before running these commands.

## Layout

```text
experiments/
  benchmark.py            # solver comparison, writes canonical CSV rows
  run_heatmap.py          # Snacks test-error heatmap grid
  run_scaling.py          # solver runtime scaling study
  run_diagnostics.py      # per-stage convergence diagnostics
  run_higgs_auc.py        # targeted HIGGS 1 - AUC evaluation
  generate_tables.py      # LaTeX tables from raw_results.csv
  summarize_results.py    # markdown summary tables
  plot_results.py         # paper figures
  __init__.py             # registers the BuRd matplotlib colormap
  lib/                    # shared experiment-only helpers
  data/                   # downloaded LIBSVM data, ignored by git
  results/                # CSVs, summaries, and run metadata
  figures/                # generated PDFs/PNGs
```

The top-level scripts are the stable entry points for the current experiment
pipeline. Shared code belongs in `experiments/lib/`; outputs belong in
`data/`, `results/`, or `figures/`.

## Pipeline

```bash
mkdir -p experiments/results experiments/figures /tmp/mplconfig && \
  PYTHONPATH=src .venv/bin/python experiments/benchmark.py \
    --dataset all && \
  PYTHONPATH=src .venv/bin/python experiments/run_heatmap.py && \
  PYTHONPATH=src .venv/bin/python experiments/run_scaling.py && \
  PYTHONPATH=src .venv/bin/python experiments/run_diagnostics.py && \
  PYTHONPATH=src .venv/bin/python experiments/generate_tables.py && \
  PYTHONPATH=src .venv/bin/python experiments/summarize_results.py && \
  MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=src .venv/bin/python experiments/plot_results.py
```

Dataset files are expected in LIBSVM format under `experiments/data/`. The
registry and split policy live in `experiments/lib/datasets.py`.

## Smoke runs

```bash
PYTHONPATH=src uv run --no-dev --extra repro python experiments/benchmark.py --dataset synthetic --smoke
PYTHONPATH=src uv run --no-dev --extra repro python experiments/run_heatmap.py --smoke
PYTHONPATH=src uv run --no-dev --extra repro python experiments/run_scaling.py --smoke
PYTHONPATH=src uv run --no-dev --extra repro python experiments/run_l1_svm.py --smoke
```

The first three write rows into the canonical CSV at:

```
experiments/results/raw_results.csv
```

`run_l1_svm.py` writes to `experiments/results/l1_svm.csv`.

Re-running appends; rows are tagged with a `run_id` and a per-run metadata file
`experiments/results/run_metadata_<run_id>.json` records the command, package
versions, git commit, and platform.

## Full Benchmark

```bash
PYTHONPATH=src uv run --no-dev --extra repro python experiments/benchmark.py --dataset covtype.binary
PYTHONPATH=src uv run --no-dev --extra repro python experiments/benchmark.py --dataset all
```

### Large Nyström runs

```bash
PYTHONPATH=src .venv/bin/python experiments/benchmark.py \
  --dataset HIGGS \
  --solvers Snacks \
  --lam-grid 1e-4

PYTHONPATH=src .venv/bin/python experiments/benchmark.py \
  --dataset all \
  --solvers Snacks
```

Raw arrays and shared Nyström features are float32 by default. The default
`--m-grid auto` rounds the theory-scaled Nyström ranks to a 1-2-5 sequence and
keeps only configs whose estimated peak memory fits `--max-peak-gb` (24 GiB by
default, with a 1.1x safety factor).
In the benchmark script, `--dataset all` now means the four active benchmark
datasets: `mushrooms`, `a1a`, `madelon`, and `news20binary`.
Large datasets whose validation-selected Snacks accuracy in
`experiments/results/raw_results.csv` is below the poor-accuracy threshold use
the largest selected-solver-feasible round `m` instead of the smaller
theory-scaled sweep. Solver rows whose own estimated working memory exceeds the
cap are skipped individually.
When `--lam-grid` is left on `auto`, each dataset uses the lambda selected from
the archived raw results instead of sweeping a lambda grid.
Adjust `--max-peak-gb` or `--peak-safety-factor` only when the machine
has enough free RAM for the raw Nyström arrays, sklearn's solver-side float64
copies, and allocator/BLAS overhead.
Snacks is float32-native end to end, including its Cython inner-stage kernels,
solver weights, labels, and Nyström feature matrices.

## Figures

```bash
PYTHONPATH=src uv run --no-dev --extra repro python experiments/plot_results.py
```

Reads `experiments/results/raw_results.csv` and writes PDFs/PNGs into
`experiments/figures/`.

| File | Description |
|------|-------------|
| `convergence.pdf` | Per-stage train/test accuracy vs elapsed solver time |
| `heatmap_<dataset>.pdf` | Snacks test-error heatmap over `(m, lambda <= 5e-3)` |
| `training_problem_scaling.pdf` | Synthetic training time vs training problem dimension `n x m` |
| `fixed_n_scaling.pdf` | Synthetic fixed-`N` Snacks vs Pegasos training time vs Nyström rank |

Additional scaling figures are generated by:

```bash
PYTHONPATH=src uv run --no-dev --extra repro python experiments/run_training_problem_scaling.py --plot-only
PYTHONPATH=src uv run --no-dev --extra repro python experiments/run_scaling.py --synthetic-fixed-n --plot-only
```

## L1-Regularized SVM

```bash
PYTHONPATH=src uv run --no-dev --extra repro python experiments/run_l1_svm.py --smoke
PYTHONPATH=src uv run --no-dev --extra repro python experiments/run_l1_svm.py --dataset mushrooms --m 200
PYTHONPATH=src uv run --no-dev --extra repro python experiments/plot_l1_svm_loss.py --n-train 200000 --m 300
PYTHONPATH=src uv run --no-dev --extra repro python experiments/plot_l1_svm_loss.py --dataset SUSY --max-train 500000 --m 300
PYTHONPATH=src uv run --no-dev --extra repro python experiments/plot_l1_svm_loss.py --n-train 300000 --m 800 --condition-number 1000000 --label-noise 0.2 --time-budget 30 --paper-schedule --pegasos-eta0 30000 --skip-liblinear --figure-stem l1_svm_loss_synthetic_paper_30s
PYTHONPATH=src uv run --no-dev --extra repro python experiments/plot_l1_svm_loss_multiseed.py --n-seeds 20
```

This targeted script compares Snacks, Pegasos, and LibLinear on shared Nyström
features with an L1 regularizer. Snacks and the plotted Pegasos trace both use
explicit stochastic subgradients for the hinge loss and the L1 term, with
`sign(0)=0` and no proximal/soft-thresholding step. sklearn's LibLinear path
only supports L1 with squared hinge, so its row is reported as a near-baseline
and the CSV records both hinge-L1 and squared-hinge-L1 diagnostics. Results are
written to `experiments/results/l1_svm.csv`.

`plot_l1_svm_loss.py` generates either a large ill-conditioned synthetic SVM
problem or a shared Nyström embedding of a registered real dataset such as
`SUSY`, then plots training hinge loss against solver-only training time on
log-log axes. Nyström fitting and feature transformation are printed but
excluded from the plotted x-axis. The CSV contains the raw stochastic traces;
the figure plots best-so-far loss by default, since stochastic subgradient
iterates are not monotone. Pass `--raw-trace` to plot the raw iterates instead.
Pass `--plot-only --gap-to-best` to redraw an existing CSV as a best-observed
loss gap, which makes convergence near the SUSY loss floor easier to inspect.
By default, Snacks and Pegasos run until they pass a `2` second solver-time
budget or exhaust their maximum iteration budget; the L1 Snacks schedule keeps
`beta_decay=2`. The synthetic path uses the
ill-conditioned synthetic schedule, while real datasets use the SUSY-tuned
defaults `m_inner0=192` and `beta0_scale=0.5`. The default synthetic figure is
`experiments/figures/l1_svm_loss.pdf`, while `--dataset SUSY` writes
`experiments/figures/l1_svm_loss_susy.pdf` and
`experiments/results/l1_svm_loss_trace_susy.csv` unless explicit output paths
are passed.

`--paper-schedule` fixes the schedule from Xu et al.: SSG/Pegasos uses
`eta_t = eta0 / sqrt(t)`, RASSG restarts every `5` stages, the inner horizon
grows by `1.15` at each restart, and Snacks' first-stage `beta` is computed
from the single tuned `eta0`.

`plot_l1_svm_loss_multiseed.py` repeats the same paper-schedule hard synthetic
experiment across seeds and plots mean best-so-far hinge loss with `+/-1`
standard-deviation ribbons. It writes after each seed and supports
`--reuse-existing` and `--plot-only` for resume/redraw workflows.

## Targeted Metrics

```bash
PYTHONPATH=src .venv/bin/python experiments/run_higgs_auc.py --solvers Snacks
PYTHONPATH=src .venv/bin/python experiments/run_higgs_auc.py --solvers Snacks --m 100
PYTHONPATH=src .venv/bin/python experiments/run_higgs_auc.py --solvers Snacks --m 500 --feature-dtype float32
```

The HIGGS `1 - AUC` value requires test-set decision scores and cannot be
recovered from `raw_results.csv`, which stores accuracies only. The targeted
script reruns HIGGS at the validation-selected Snacks benchmark configuration
and writes `experiments/results/higgs_auc.csv`. Larger `m` can be tested with
`--m`; the script estimates dense Nyström memory and refuses configs above
`--max-peak-gb` unless that budget is explicitly changed. `--raw-dtype` and
`--feature-dtype` are float32-only.

## CSV schema

The canonical row schema is defined in `experiments/lib/common.py` (`FIELDS`).
Empty fields mean "not applicable to this row".

Timing fields:

- `train_time_solver`, `predict_time_solver` — time of the solver only, on
  already-embedded Nyström features. For Snacks this is the compiled optimizer
  path only; validation selection is reported separately in row notes. Warmup
  calls are excluded.
- `nystrom_fit_time` — time of the column eigendecomposition (one per
  config/seed).
- `nystrom_transform_train_time`, `..._val_time`, `..._test_time` — feature
  transform times for each split.
- `total_train_time` — Nyström fit + train transform (+ val transform for
  ASSG-r, used for early stopping) + solver fit. End-to-end training cost.
- `total_predict_time` — test transform + solver predict.
- `total_time` — full pipeline including all of the above.

For shared-feature solvers (ASSG-r, LibLinear, Pegasos), the Nyström pipeline
cost is amortized across all three rows of a (lam, m, seed) configuration.
For `sklearn-Nystrom`, only `total_train_time`, `total_predict_time`, and
`total_time` are populated, since its embedding is not separable from its
fit/predict path.

## Defaults

`benchmark.py`:

| Parameter | Value |
|-----------|-------|
| lambda | fixed per dataset from archived validation results (`1e-4` for the current benchmark datasets) |
| raw dtype | `float32` |
| feature dtype | `float32` |
| m grid | `auto` by default: rounded memory-capped grid around `ceil(n_train^0.5 * log(n_train / 0.05))`; poor large datasets use the largest selected-solver-feasible rank |
| m train-feature cap | 1024 MB for dense `Z_train` in the theory-scaled grid |
| peak memory cap | 24 GiB with a 1.1x safety factor |
| n_stages | 20 |
| Snacks restarts | 4 |
| Snacks stages per restart | 10 |
| Snacks m_inner0 | 512 |
| Seeds | 0, 1, 2 |

`run_diagnostics.py` defaults to the Snacks validation-selected `(lambda, m)`
configuration from `experiments/results/raw_results.csv` and plots convergence
on `HIGGS`, `YearPredictionMSD`, `ijcnn1`, and `w8a`. Use `--lam` or `--m` to
force a fixed diagnostic configuration.

The `m` default is a transparent heuristic based on Proposition 1 / Theorem 3
of Della Vecchia et al. (2021): under polynomial eigendecay and leverage-score
sampling, a subspace size on the order of `n^p log(n / delta)` can preserve the
statistical rate, with `p=0.5` used here as the default reference case. This
code still uses uniform column sampling, so the auto grid should be read as a
theory-scaled sweep, not as a formal guarantee. Override it with
`--m-grid 100,500,1000`; tune the cap with `--max-feature-mb`.
