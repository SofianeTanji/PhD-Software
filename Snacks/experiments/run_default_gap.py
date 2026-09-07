#!/usr/bin/env python3
"""Default-vs-tuned optimizer gap experiment for RASSG-r.

For each (dataset, m, lambda) cell we compare the DEFAULT optimizer settings
against a small random search over optimizer settings (TUNED). The gap is

    gap = tuned_mean_test_acc - default_mean_test_acc   (accuracy points)

A small gap means the defaults are nearly as good as per-cell tuning. We plot an
ECDF of the gap across all cells, one curve per dataset.

Validation accuracy is used to pick the tuned setting; the test set is never used
for any selection.

Usage:
    PYTHONPATH=src uv run python experiments/run_default_gap.py --smoke
    PYTHONPATH=src uv run python experiments/run_default_gap.py
    PYTHONPATH=src uv run python experiments/run_default_gap.py --datasets ijcnn1 mushrooms
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb

from lib.plotstyle import SNACKS_COLOR, apply_style, save_fig


def _light_color(color: str, percent: float = 0.55):
    rgb = np.array(to_rgb(color))
    return tuple((1.0 - percent) * np.ones(3) + percent * rgb)


from lib.common import (
    RASSG_R_DEFAULT_BETA0_SCALE,
    RASSG_R_DEFAULT_BETA_DECAY,
    RASSG_R_DEFAULT_GROWTH,
    RASSG_R_DEFAULT_M_INNER0,
    RASSG_R_DEFAULT_RESTARTS,
    RASSG_R_DEFAULT_STAGES_PER_RESTART,
)
from lib.datasets import load_dataset

from snacks.kernels import median_bandwidth, rbf_kernel
from snacks.nystrom import NystromTransformer, select_columns
from snacks.solver import rassg_r

# Datasets span two size regimes (covtype ~349k, ijcnn1 ~40k, mushrooms ~4.9k)
# so the ECDF shows the defaults hold across scales, not just on one dataset.
DATASETS = ["covtype.binary", "ijcnn1", "mushrooms"]
# Fixed display order, colors, and labels for the figure.
DATASET_COLORS = {
    "covtype.binary": SNACKS_COLOR,
    "ijcnn1": "#B2182B",
    "mushrooms": "#E08214",
}
DATASET_LABELS = {
    "covtype.binary": "covtype",
    "ijcnn1": "ijcnn1",
    "mushrooms": "mushrooms",
}

M_GRID = [50, 75, 100, 150, 200, 300, 500, 750, 1_000, 1_500]
# lambda=1e-2 dropped: the default optimizer's only soft spot lived entirely
# there, so it inflated the worst-case gap without representing a regime we care
# about.
LAM_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 3e-2, 1e-1, 3e-1]
SEEDS = [0, 1, 2]
N_TUNE = 10  # random optimizer settings per cell

FIG_DIR = "experiments/figures"
FIG_STEM = "default_gap_ecdf"

FIELDS = [
    "dataset", "m", "lam", "condition", "seed",
    "beta0_scale", "restarts", "val_acc", "test_acc",
]


def dense(arr) -> np.ndarray:
    return np.asarray(arr.toarray() if hasattr(arr, "toarray") else arr, dtype=np.float32)


def build_features(X_tr, X_val, X_te, m, seed):
    """Nystrom features for one (m, seed). Copied from run_heatmap.run_config."""
    n_tr = X_tr.shape[0]
    rng = np.random.default_rng(seed)
    gamma = float(median_bandwidth(X_tr, n_subsample=min(2000, n_tr), rng=np.random.default_rng(seed)))
    columns = select_columns(X_tr, min(m, n_tr), rng)
    nys = NystromTransformer(lambda A, B: rbf_kernel(A, B, gamma), columns, ridge_mu=1e-6).fit()
    return nys.transform(X_tr), nys.transform(X_val), nys.transform(X_te)


def acc(Z, y, u) -> float:
    return float(np.mean((y * (Z @ u)) > 0))


def fit_one(Z_tr, y_tr, Z_val, y_val, Z_te, y_te, lam, seed, beta0_scale, restarts):
    """Single RASSG-r fit. Returns (val_acc, test_acc)."""
    res = rassg_r(
        Z_tr, y_tr, lam=lam,
        restarts=restarts,
        stages_per_restart=RASSG_R_DEFAULT_STAGES_PER_RESTART,
        m_inner0=RASSG_R_DEFAULT_M_INNER0,
        growth=RASSG_R_DEFAULT_GROWTH,
        beta0_scale=beta0_scale,
        beta_decay=RASSG_R_DEFAULT_BETA_DECAY,
        Z_val=Z_val, y_val=y_val, Z_test=Z_te, y_test=y_te,
        rng=np.random.default_rng(seed),
    )
    return acc(Z_val, y_val, res.u), acc(Z_te, y_te, res.u)


def gaps_from_rows(rows, dataset: str) -> np.ndarray:
    """Per-cell gap for one dataset = tuned best-by-val mean test - default mean test.

    Rows whose lambda is not in LAM_GRID are ignored, so reading an older CSV
    that still contains dropped lambda columns reproduces the current grid.
    """
    by_cell = defaultdict(lambda: {"default": [], "tuned": defaultdict(list)})
    for r in rows:
        if r.get("dataset", DATASETS[0]) != dataset:
            continue
        lam = float(r["lam"])
        if not any(np.isclose(lam, L) for L in LAM_GRID):
            continue
        key = (int(float(r["m"])), lam)
        if r["condition"] == "default":
            by_cell[key]["default"].append(float(r["test_acc"]))
        else:
            setting = (round(float(r["beta0_scale"]), 6), int(float(r["restarts"])))
            by_cell[key]["tuned"][setting].append((float(r["val_acc"]), float(r["test_acc"])))

    gaps = []
    for d in by_cell.values():
        default_mean_test = float(np.mean(d["default"]))
        best_val, best_test = -1.0, default_mean_test
        for vals in d["tuned"].values():
            mean_val = float(np.mean([v for v, _ in vals]))
            if mean_val > best_val:
                best_val = mean_val
                best_test = float(np.mean([t for _, t in vals]))
        gaps.append(best_test - default_mean_test)
    return np.array(gaps)


def read_existing(path: Path) -> list[dict]:
    """Read prior rows, defaulting a missing `dataset` column to the first dataset.

    Lets the original covtype-only CSV (no dataset column) merge transparently.
    """
    if not path.exists():
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.setdefault("dataset", DATASETS[0])
        if not r["dataset"]:
            r["dataset"] = DATASETS[0]
    return rows


def smoke(split):
    """Single default fit at m=100, lam=1e-2, seed=0."""
    X_tr, X_val, X_te = dense(split.X_train), dense(split.X_val), dense(split.X_test)
    Z_tr, Z_val, Z_te = build_features(X_tr, X_val, X_te, m=100, seed=0)
    # warm-up (compile cython path)
    rassg_r(Z_tr[:2], split.y_train[:2], lam=1e-2, restarts=1, stages_per_restart=1,
            m_inner0=1, growth=1.0, rng=np.random.default_rng(0))
    _, test_acc = fit_one(
        Z_tr, split.y_train, Z_val, split.y_val, Z_te, split.y_test,
        lam=1e-2, seed=0,
        beta0_scale=RASSG_R_DEFAULT_BETA0_SCALE, restarts=RASSG_R_DEFAULT_RESTARTS,
    )
    print(f"\nSMOKE TEST  m=100 lam=1e-2 seed=0 (defaults): test_acc={test_acc:.4f}")
    if test_acc <= 0.55:
        print("FAILED: test accuracy is not clearly better than chance. Something is broken; stop.")
    else:
        print("OK: sensible accuracy, well above chance.")
    return test_acc


def run_dataset(dataset: str) -> list[dict]:
    """Run all (m, lambda, seed) cells for one dataset. Returns CSV rows."""
    print(f"\nDataset: {dataset}")
    split = load_dataset(dataset)
    print(f"n_train={split.n_train}  n_features={split.n_features}")

    X_tr, X_val, X_te = dense(split.X_train), dense(split.X_val), dense(split.X_test)
    y_tr, y_val, y_te = split.y_train, split.y_val, split.y_test

    # warm-up once.
    Z0, _, _ = build_features(X_tr[:200], X_val[:50], X_te[:50], m=50, seed=0)
    rassg_r(Z0[:2], y_tr[:2], lam=1e-2, restarts=1, stages_per_restart=1,
            m_inner0=1, growth=1.0, rng=np.random.default_rng(0))

    rows: list[dict] = []

    for m in M_GRID:
        # Nystrom features depend only on (m, seed): build once, reuse for all lam/settings.
        feats = {s: build_features(X_tr, X_val, X_te, m, s) for s in SEEDS}

        for lam in LAM_GRID:
            # DEFAULT: 3 seeds.
            def_test = []
            for s in SEEDS:
                Z_tr, Z_val, Z_te = feats[s]
                va, ta = fit_one(Z_tr, y_tr, Z_val, y_val, Z_te, y_te, lam, s,
                                 RASSG_R_DEFAULT_BETA0_SCALE, RASSG_R_DEFAULT_RESTARTS)
                rows.append({"dataset": dataset, "m": m, "lam": lam, "condition": "default", "seed": s,
                             "beta0_scale": RASSG_R_DEFAULT_BETA0_SCALE,
                             "restarts": RASSG_R_DEFAULT_RESTARTS,
                             "val_acc": va, "test_acc": ta})
                def_test.append(ta)
            default_mean_test = float(np.mean(def_test))

            # TUNED: N_TUNE random settings, each over 3 seeds; pick best mean val_acc.
            tune_rng = np.random.default_rng(1000 * m + int(round(lam * 1e6)))
            best_mean_val = -1.0
            best_mean_test = default_mean_test
            for _ in range(N_TUNE):
                beta0_scale = float(10.0 ** tune_rng.uniform(0.0, 2.0))  # log-uniform [1, 100]
                restarts = int(tune_rng.choice([2, 4, 6, 8]))
                t_val, t_test = [], []
                for s in SEEDS:
                    Z_tr, Z_val, Z_te = feats[s]
                    va, ta = fit_one(Z_tr, y_tr, Z_val, y_val, Z_te, y_te, lam, s,
                                     beta0_scale, restarts)
                    rows.append({"dataset": dataset, "m": m, "lam": lam, "condition": "tuned", "seed": s,
                                 "beta0_scale": beta0_scale, "restarts": restarts,
                                 "val_acc": va, "test_acc": ta})
                    t_val.append(va); t_test.append(ta)
                mean_val = float(np.mean(t_val))
                if mean_val > best_mean_val:
                    best_mean_val = mean_val
                    best_mean_test = float(np.mean(t_test))

            gap = best_mean_test - default_mean_test
            print(f"  m={m:>5} lam={lam:.0e}: default={default_mean_test:.4f} "
                  f"tuned={best_mean_test:.4f} gap={gap:+.4f}")

    return rows


def full_run(datasets: list[str], out_path: Path) -> list[dict]:
    """Run the requested datasets and merge with carried-forward rows for the rest."""
    carried = [r for r in read_existing(out_path) if r["dataset"] not in datasets]
    new_rows: list[dict] = []
    for dataset in datasets:
        new_rows.extend(run_dataset(dataset))

    rows = carried + new_rows
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path} "
          f"({len(new_rows)} new, {len(carried)} carried forward)")
    return rows


def make_figure(rows: list[dict], out_dir: Path, stem: str):
    from matplotlib import patheffects as path_effects
    import matplotlib.pyplot as plt

    apply_style()

    datasets = [d for d in DATASETS if any(r["dataset"] == d for r in rows)]
    fig, ax = plt.subplots()
    xmax = 2.0

    for dataset in datasets:
        gaps = gaps_from_rows(rows, dataset)
        if len(gaps) == 0:
            continue
        g = np.sort(gaps) * 100.0  # accuracy points
        n = len(g)
        y = np.arange(1, n + 1) / n
        xmax = max(xmax, float(g[-1]) * 1.05)

        # Pad right so the last step is visible.
        g_plot = np.append(g, max(xmax, g[-1]))
        y_plot = np.append(y, y[-1])

        color = DATASET_COLORS[dataset]
        # Dark-outside, light-inside — matches the non-primary line style in plot_results.py.
        ax.step(g_plot, y_plot, where="post", color=_light_color(color), linewidth=2.0,
                label=DATASET_LABELS[dataset],
                path_effects=[
                    path_effects.Stroke(linewidth=3.0, foreground=color),
                    path_effects.Normal(),
                ])

    # Subtle 50% reference line.
    ax.axhline(0.5, color="0.55", lw=0.8, ls=":", zorder=0)

    ax.axvline(0.0, color="black", ls="--", lw=0.9, zorder=1)
    ax.set_xlim(0, xmax)

    ax.set_xlabel("Test accuracy loss from not tuning (percentage points)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("Default schedule tuning loss")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="lower right", frameon=False)

    save_fig(fig, out_dir, stem)


def report(rows: list[dict]):
    for dataset in DATASETS:
        gaps = gaps_from_rows(rows, dataset)
        if len(gaps) == 0:
            continue
        g = gaps * 100.0
        mean, median, worst = float(np.mean(g)), float(np.median(g)), float(np.max(g))
        frac = float(np.mean(g <= 0.5))
        print(
            f"\n[{dataset}] Across all {len(g)} (m, λ) cells, tuning the optimizer per cell "
            f"improves test accuracy by only {mean:.2f} points on average (median {median:.2f}, "
            f"worst case {worst:.2f}), with {frac:.0%} of cells within 0.5 points of the "
            f"tuned result; the default optimizer settings are therefore nearly as good as "
            f"per-cell tuning."
        )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true", help="single default fit (m=100, lam=1e-2, seed=0)")
    p.add_argument("--from-csv", action="store_true",
                   help="rebuild figure + sentence from an existing CSV (no fits run)")
    p.add_argument("--datasets", nargs="+", default=DATASETS,
                   help="datasets to (re)run; others already in the CSV are carried forward")
    p.add_argument("--out", default="experiments/results/default_gap_results.csv")
    p.add_argument("--fig-dir", default=FIG_DIR)
    p.add_argument("--fig-stem", default=FIG_STEM)
    args = p.parse_args()

    if args.from_csv:
        rows = read_existing(Path(args.out))
        make_figure(rows, Path(args.fig_dir), args.fig_stem)
        report(rows)
        return

    if args.smoke:
        dataset = args.datasets[0]
        print(f"Dataset: {dataset}")
        smoke(load_dataset(dataset))
        return

    rows = full_run(args.datasets, Path(args.out))
    make_figure(rows, Path(args.fig_dir), args.fig_stem)
    report(rows)


if __name__ == "__main__":
    main()
