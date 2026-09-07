#!/usr/bin/env python3
"""Scaling experiment: solver training time vs number of Nyström columns m.

Fixes n to the full dataset, varies m over a grid, and runs both RASSG-r and
Pegasos on the *same* Nystrom features (same columns per seed) to isolate
the effect of m on solver cost.

Usage:
    PYTHONPATH=src uv run python experiments/run_scaling.py
    PYTHONPATH=src uv run python experiments/run_scaling.py --dataset ijcnn1
    PYTHONPATH=src uv run python experiments/run_scaling.py --smoke
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as path_effects
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import experiments  # registers BuRd colormap

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

DEFAULT_DATASET = "covtype.binary"
M_GRID = [50, 100, 200, 500, 1_000, 2_000]
N_SEEDS = 3
LAM = 1e-3
N_STAGES = 20
STYLE_PATH = Path(__file__).parent.parent / "custom_style.mplstyle"
DEFAULT_SYNTHETIC_OUT = "experiments/results/fixed_n_scaling_results.csv"
DEFAULT_SYNTHETIC_FIGURE = "experiments/figures/fixed_n_scaling.pdf"
PLOT_COLORS = {"RASSG-r": "#194F8C", "Pegasos": "#EB9486"}
PLOT_LABELS = {"RASSG-r": "Snacks", "Pegasos": "Pegasos"}

FIELDS = [
    "dataset",
    "solver",
    "seed",
    "lam",
    "n_train",
    "m",
    "nystrom_fit_time",
    "nystrom_transform_time",
    "solver_train_time",
    "n_stages_run",
    "stop_reason",
]


def dense(arr) -> np.ndarray:
    return np.asarray(
        arr.toarray() if hasattr(arr, "toarray") else arr, dtype=np.float32
    )


def parse_int_grid(value: str) -> list[int]:
    grid = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not grid or any(v <= 0 for v in grid):
        raise ValueError("m grid must contain positive integers.")
    return grid


def geom_int_grid(lo: int, hi: int, count: int) -> list[int]:
    vals = [int(round(v)) for v in np.geomspace(lo, hi, count)]
    out: list[int] = []
    for value in vals:
        if value not in out:
            out.append(value)
    return out


def light_color(color: str, percent: float = 0.55) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb

    rgb = np.array(to_rgb(color))
    return tuple((1.0 - percent) * np.ones(3) + percent * rgb)


def run_m_config(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    dataset: str,
    lam: float,
    m: int,
    seed: int,
    n_stages: int,
) -> list[dict]:
    """Run RASSG-r and Pegasos on identical Nystrom features for one (m, seed)."""
    n_tr = len(y_tr)
    rng = np.random.default_rng(seed)

    gamma = float(
        median_bandwidth(
            X_tr, n_subsample=min(2000, n_tr), rng=np.random.default_rng(seed)
        )
    )
    columns = select_columns(X_tr, min(m, n_tr), rng)

    t0 = time.perf_counter()
    nys = NystromTransformer(
        lambda A, B: rbf_kernel(A, B, gamma), columns, ridge_mu=1e-6
    ).fit()
    nystrom_fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    Z_tr = nys.transform(X_tr)
    Z_val = nys.transform(X_val)
    nystrom_transform_time = time.perf_counter() - t0

    base = dict(
        dataset=dataset,
        seed=seed,
        lam=lam,
        n_train=n_tr,
        m=min(m, n_tr),
        nystrom_fit_time=nystrom_fit_time,
        nystrom_transform_time=nystrom_transform_time,
    )

    rows: list[dict] = []

    # --- RASSG-r ---
    rassg_r(
        Z_tr[:2],
        y_tr[:2],
        lam=lam,
        restarts=1,
        stages_per_restart=1,
        m_inner0=1,
        growth=1.0,
        rng=np.random.default_rng(0),
    )
    t0 = time.perf_counter()
    res = rassg_r(
        Z_tr,
        y_tr,
        lam=lam,
        restarts=RASSG_R_DEFAULT_RESTARTS,
        stages_per_restart=RASSG_R_DEFAULT_STAGES_PER_RESTART,
        m_inner0=RASSG_R_DEFAULT_M_INNER0,
        growth=RASSG_R_DEFAULT_GROWTH,
        beta0_scale=RASSG_R_DEFAULT_BETA0_SCALE,
        beta_decay=RASSG_R_DEFAULT_BETA_DECAY,
        Z_val=Z_val,
        y_val=y_val,
        rng=np.random.default_rng(seed),
    )
    rows.append(
        {
            **base,
            "solver": "RASSG-r",
            "solver_train_time": time.perf_counter() - t0,
            "n_stages_run": res.n_stages_run,
            "stop_reason": res.stop_reason,
        }
    )

    # --- Pegasos ---
    clf = SGDClassifier(
        loss="hinge",
        penalty="l2",
        alpha=lam,
        learning_rate="optimal",
        fit_intercept=False,
        max_iter=n_stages,
        shuffle=True,
        random_state=seed,
    )
    t0 = time.perf_counter()
    clf.fit(Z_tr, y_tr)
    rows.append(
        {
            **base,
            "solver": "Pegasos",
            "solver_train_time": time.perf_counter() - t0,
            "n_stages_run": n_stages,
            "stop_reason": "max_iter",
        }
    )

    return rows


def make_synthetic_split(n_train: int, n_val: int, n_features: int, seed: int):
    X, y = make_classification(
        n_samples=n_train + n_val,
        n_features=n_features,
        n_informative=max(2, int(n_features * 0.6)),
        n_redundant=max(0, int(n_features * 0.2)),
        n_clusters_per_class=2,
        class_sep=1.2,
        flip_y=0.03,
        random_state=seed,
    )
    X = X.astype(np.float32, copy=False)
    y = np.where(y > 0, 1.0, -1.0).astype(np.float32)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def write_rows(out_path: Path, rows: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot_fixed_n_scaling(rows: list[dict], figure_path: Path) -> None:
    plt.style.use(str(STYLE_PATH))
    by_solver_m: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        solver = row.get("solver", "")
        if solver not in PLOT_COLORS:
            continue
        by_solver_m[(solver, int(row["m"]))].append(float(row["solver_train_time"]))

    fig, ax = plt.subplots(figsize=(7, 4))
    n_train = int(float(rows[0]["n_train"])) if rows else 0
    ax.set_title(rf"Fixed-$N$ scaling -- Synthetic ($N={n_train:,}$)")

    for solver in ("RASSG-r", "Pegasos"):
        ms = sorted(m for s, m in by_solver_m if s == solver)
        if not ms:
            continue
        color = PLOT_COLORS[solver]
        lc = light_color(color)
        ys = [mean(by_solver_m[(solver, m)]) for m in ms]
        if any(len(by_solver_m[(solver, m)]) > 1 for m in ms):
            lo = []
            hi = []
            for m, y in zip(ms, ys):
                vals = by_solver_m[(solver, m)]
                sd = stdev(vals) if len(vals) > 1 else 0.0
                lo.append(max(1e-12, y - sd))
                hi.append(y + sd)
            ax.fill_between(ms, lo, hi, color=color, alpha=0.15, zorder=2)
        ax.scatter(
            ms,
            ys,
            s=12,
            color=lc,
            edgecolors=color,
            linewidths=0.3,
            alpha=0.65,
            zorder=3,
        )
        ax.plot(
            ms,
            ys,
            color=lc,
            linewidth=2.0,
            label=PLOT_LABELS[solver],
            zorder=6,
            path_effects=[
                path_effects.Stroke(linewidth=3.2, foreground=color),
                path_effects.Normal(),
            ],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Nystrom columns")
    ax.set_ylabel("Training time (s)")
    ax.grid(True, which="major", color="#cfcfcf", linestyle=":", linewidth=0.6)
    ax.grid(True, which="minor", color="#dedede", linestyle=":", linewidth=0.35)
    ax.tick_params(which="both", width=1.2)
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.legend(loc="upper left")
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path)
    plt.close(fig)
    print(f"Saved {figure_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--lam", type=float, default=LAM)
    p.add_argument("--n-seeds", type=int, default=N_SEEDS)
    p.add_argument("--smoke", action="store_true", help="tiny m grid, 2 seeds")
    p.add_argument("--out", default="experiments/results/scaling_results.csv")
    p.add_argument("--synthetic-fixed-n", action="store_true")
    p.add_argument("--n-train", type=int, default=1_000_000)
    p.add_argument("--n-val", type=int, default=10_000)
    p.add_argument("--n-features", type=int, default=40)
    p.add_argument("--m-grid", default="")
    p.add_argument("--m-min", type=int, default=10)
    p.add_argument("--m-max", type=int, default=1000)
    p.add_argument("--m-points", type=int, default=16)
    p.add_argument("--figure", default="")
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()

    if args.synthetic_fixed_n:
        out_path = Path(
            args.out
            if args.out != "experiments/results/scaling_results.csv"
            else DEFAULT_SYNTHETIC_OUT
        )
        figure_path = Path(args.figure or DEFAULT_SYNTHETIC_FIGURE)
        if args.plot_only:
            rows = load_rows(out_path)
            print(f"Loaded {len(rows)} rows from {out_path}")
            plot_fixed_n_scaling(rows, figure_path)
            return

        if args.m_grid:
            m_grid = parse_int_grid(args.m_grid)
        else:
            m_grid = geom_int_grid(args.m_min, args.m_max, args.m_points)
        if args.n_seeds == N_SEEDS:
            args.n_seeds = 1
        if args.smoke:
            args.n_train = 20_000
            args.n_val = 2_000
            m_grid = [10, 30, 100]
            args.n_seeds = 1
            n_stages = 3
        else:
            n_stages = N_STAGES

        print("Dataset: synthetic")
        print(f"n_train={args.n_train}  n_features={args.n_features}")
        print(f"m_grid={m_grid}  seeds=0..{args.n_seeds - 1}")
        X_tr, y_tr, X_val, y_val = make_synthetic_split(
            args.n_train,
            args.n_val,
            args.n_features,
            seed=0,
        )

        all_rows: list[dict] = []
        for m in m_grid:
            for seed in range(args.n_seeds):
                rows = run_m_config(
                    X_tr,
                    y_tr,
                    X_val,
                    y_val,
                    dataset="synthetic-fixed-n",
                    lam=args.lam,
                    m=m,
                    seed=seed,
                    n_stages=n_stages,
                )
                all_rows.extend(rows)
                for r in rows:
                    print(
                        f"  m={m:>5} seed={seed} {r['solver']:<8}: "
                        f"solver={r['solver_train_time']:.3f}s"
                    )

        write_rows(out_path, all_rows)
        print(f"\nWrote {len(all_rows)} rows to {out_path}")
        plot_fixed_n_scaling(all_rows, figure_path)
        return

    m_grid = [50, 100] if args.smoke else M_GRID
    n_seeds = 2 if args.smoke else args.n_seeds
    n_stages = 3 if args.smoke else N_STAGES

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset}")
    split = load_dataset(args.dataset)
    X_tr = dense(split.X_train)
    X_val = dense(split.X_val)
    print(f"n_train={split.n_train}  n_features={split.n_features}")
    print(f"m_grid={m_grid}  seeds=0..{n_seeds - 1}")

    all_rows: list[dict] = []
    for m in m_grid:
        for seed in range(n_seeds):
            try:
                rows = run_m_config(
                    X_tr,
                    split.y_train,
                    X_val,
                    split.y_val,
                    dataset=args.dataset,
                    lam=args.lam,
                    m=m,
                    seed=seed,
                    n_stages=n_stages,
                )
                all_rows.extend(rows)
                for r in rows:
                    print(
                        f"  m={m:>5} seed={seed} {r['solver']:<8}: "
                        f"solver={r['solver_train_time']:.3f}s"
                    )
            except Exception as e:
                print(f"  m={m} seed={seed}: FAILED: {e}")

    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
