#!/usr/bin/env python3
"""Training-time scaling over subsampled synthetic Nyström problems.

The figure keeps the Playground stability aesthetic, but the x-axis is now the
training problem dimension n * m: number of training points times number of
Nyström columns. Both n and m vary across the grid.

Usage:
    PYTHONPATH=src uv run python experiments/run_training_problem_scaling.py
    PYTHONPATH=src uv run python experiments/run_training_problem_scaling.py --smoke
"""

from __future__ import annotations

import argparse
import csv
import gc
import time
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as path_effects
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import experiments  # registers BuRd colormap

from snacks.kernels import median_bandwidth, rbf_kernel
from snacks.nystrom import NystromTransformer, select_columns
from snacks.solver import rassg_r


RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
DEFAULT_CSV = RESULTS_DIR / "training_problem_scaling.csv"
DEFAULT_FIGURE = FIGURES_DIR / "training_problem_scaling.pdf"
STYLE_PATH = Path(__file__).parent.parent / "custom_style.mplstyle"
DEFAULT_EXCLUDE_DIMENSIONS = "1000,53067,78318,174748"

SOLVERS = ("RASSG-r", "LibLinear", "Pegasos", "sklearn-Nystrom")
SOLVER_LABELS = {
    "RASSG-r": "Snacks",
    "LibLinear": "LibLinear",
    "Pegasos": "Pegasos",
    "sklearn-Nystrom": "sklearn-Nystrom",
}
SOLVER_COLORS = {
    "RASSG-r": "#194F8C",
    "LibLinear": "#A0B78F",
    "Pegasos": "#EB9486",
    "sklearn-Nystrom": "#A53860",
}


def _light_color(color: str, percent: float = 0.55) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb

    rgb = np.array(to_rgb(color))
    return tuple((1.0 - percent) * np.ones(3) + percent * rgb)


def _parse_ints(value: str) -> list[int]:
    out = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise ValueError("grid must contain at least one integer")
    if any(x <= 0 for x in out):
        raise ValueError("grid values must be positive")
    return out


def _rounded_geomspace(lo: int, hi: int, count: int, step: int) -> list[int]:
    raw = np.geomspace(lo, hi, count)
    return [max(step, int(round(x / step) * step)) for x in raw]


def _problem_grid(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.n_grid:
        n_grid = _parse_ints(args.n_grid)
    else:
        n_grid = _rounded_geomspace(args.n_min, args.n_max, args.grid_points, step=1)

    if args.m_grid:
        m_grid = _parse_ints(args.m_grid)
    else:
        m_grid = _rounded_geomspace(args.m_min, args.m_max, len(n_grid), step=1)

    if len(n_grid) != len(m_grid):
        raise ValueError("--n-grid and --m-grid must have the same length")

    grid = [(n, min(m, n)) for n, m in zip(n_grid, m_grid)]
    if len(grid) < 20 and not args.smoke and not args.n_grid:
        raise ValueError("use at least 20 grid points for the full figure")
    return grid


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "solver",
        "seed",
        "n_train",
        "m",
        "problem_dimension",
        "nystrom_fit_time",
        "nystrom_transform_train_time",
        "solver_train_time",
        "training_time",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "solver",
        "seed",
        "n_train",
        "m",
        "problem_dimension",
        "nystrom_fit_time",
        "nystrom_transform_train_time",
        "solver_train_time",
        "training_time",
    ]
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _row_key(row: dict) -> tuple[str, int, int, int]:
    return (
        row["solver"],
        int(float(row["seed"])),
        int(float(row["n_train"])),
        int(float(row["m"])),
    )


def _row(
    solver: str,
    seed: int,
    n_train: int,
    m: int,
    nystrom_fit_time: float | None,
    nystrom_transform_train_time: float | None,
    solver_train_time: float | None,
    training_time: float,
) -> dict:
    return {
        "solver": solver,
        "seed": seed,
        "n_train": n_train,
        "m": m,
        "problem_dimension": n_train * m,
        "nystrom_fit_time": ""
        if nystrom_fit_time is None
        else f"{nystrom_fit_time:.9g}",
        "nystrom_transform_train_time": (
            ""
            if nystrom_transform_train_time is None
            else f"{nystrom_transform_train_time:.9g}"
        ),
        "solver_train_time": ""
        if solver_train_time is None
        else f"{solver_train_time:.9g}",
        "training_time": f"{training_time:.9g}",
    }


def run_training_problem_scaling(args: argparse.Namespace) -> list[dict]:
    rng = np.random.default_rng(args.seed)
    X, y = make_classification(
        n_samples=args.n_population,
        n_features=args.n_features,
        n_informative=max(2, int(args.n_features * 0.6)),
        n_redundant=max(0, int(args.n_features * 0.2)),
        n_clusters_per_class=2,
        class_sep=1.2,
        flip_y=0.03,
        random_state=args.seed,
    )
    X = X.astype(np.float32, copy=False)
    y = np.where(y > 0, 1.0, -1.0).astype(np.float32)
    gamma = median_bandwidth(X, n_subsample=min(2000, args.n_population), rng=rng)
    existing_keys: set[tuple[str, int, int, int]] = set()
    if args.append and args.csv.exists():
        existing_keys = {_row_key(row) for row in _read_rows(args.csv)}

    rows: list[dict] = []
    for n_train, m in _problem_grid(args):
        for repeat in range(args.repeats):
            seed = args.seed + 10_000 * repeat + n_train + m
            pending_solvers = [
                solver
                for solver in args.solvers
                if (solver, seed, n_train, m) not in existing_keys
            ]
            if not pending_solvers:
                continue
            local_rng = np.random.default_rng(seed)
            idx = local_rng.choice(args.n_population, size=n_train, replace=False)
            X_sub = X[idx]
            y_sub = y[idx]

            if any(s in pending_solvers for s in ("RASSG-r", "LibLinear", "Pegasos")):
                columns = select_columns(X_sub, m, local_rng)
                kernel = lambda A, B: rbf_kernel(A, B, gamma=gamma)
                t0 = time.perf_counter()
                nys = NystromTransformer(kernel, columns, ridge_mu=1e-6).fit()
                nystrom_fit_time = time.perf_counter() - t0
                t0 = time.perf_counter()
                Z_sub = nys.transform(X_sub)
                nystrom_transform_train_time = time.perf_counter() - t0
                shared_feature_time = nystrom_fit_time + nystrom_transform_train_time
            else:
                Z_sub = None
                nystrom_fit_time = 0.0
                nystrom_transform_train_time = 0.0
                shared_feature_time = 0.0

            if "RASSG-r" in pending_solvers:
                t0 = time.perf_counter()
                res = rassg_r(
                    Z_sub,
                    y_sub,
                    lam=args.lam,
                    restarts=args.restarts,
                    stages_per_restart=args.stages_per_restart,
                    m_inner0=args.m_inner0,
                    growth=2.0,
                    rng=np.random.default_rng(seed),
                )
                wall_time = time.perf_counter() - t0
                solver_time = (
                    res.optimizer_time if res.optimizer_time is not None else wall_time
                )
                rows.append(
                    _row(
                        "RASSG-r",
                        seed,
                        n_train,
                        m,
                        nystrom_fit_time,
                        nystrom_transform_train_time,
                        solver_time,
                        shared_feature_time + wall_time,
                    )
                )

            if "LibLinear" in pending_solvers:
                clf = LinearSVC(
                    loss="hinge",
                    penalty="l2",
                    C=1 / (n_train * args.lam),
                    fit_intercept=False,
                    dual=True,
                    max_iter=20000,
                    random_state=seed,
                )
                t0 = time.perf_counter()
                clf.fit(Z_sub, y_sub)
                solver_time = time.perf_counter() - t0
                rows.append(
                    _row(
                        "LibLinear",
                        seed,
                        n_train,
                        m,
                        nystrom_fit_time,
                        nystrom_transform_train_time,
                        solver_time,
                        shared_feature_time + solver_time,
                    )
                )

            if "Pegasos" in pending_solvers:
                clf = SGDClassifier(
                    loss="hinge",
                    penalty="l2",
                    alpha=args.lam,
                    learning_rate="optimal",
                    fit_intercept=False,
                    max_iter=args.pegasos_epochs,
                    shuffle=True,
                    random_state=seed,
                )
                t0 = time.perf_counter()
                clf.fit(Z_sub, y_sub)
                solver_time = time.perf_counter() - t0
                rows.append(
                    _row(
                        "Pegasos",
                        seed,
                        n_train,
                        m,
                        nystrom_fit_time,
                        nystrom_transform_train_time,
                        solver_time,
                        shared_feature_time + solver_time,
                    )
                )

            if "sklearn-Nystrom" in pending_solvers:
                if Z_sub is not None:
                    del Z_sub
                    gc.collect()
                sk_nys = Nystroem(
                    kernel="rbf",
                    gamma=gamma,
                    n_components=m,
                    random_state=seed,
                )
                clf = LinearSVC(
                    loss="hinge",
                    penalty="l2",
                    C=1 / (n_train * args.lam),
                    fit_intercept=False,
                    dual=True,
                    max_iter=20000,
                    random_state=seed,
                )
                t0 = time.perf_counter()
                Z_sub_sk = sk_nys.fit_transform(X_sub).astype(np.float32, copy=False)
                clf.fit(Z_sub_sk, y_sub)
                train_time = time.perf_counter() - t0
                rows.append(
                    _row(
                        "sklearn-Nystrom",
                        seed,
                        n_train,
                        m,
                        None,
                        None,
                        None,
                        train_time,
                    )
                )

    return rows


def plot_training_problem_scaling(
    rows: list[dict],
    output_path: Path,
    *,
    exclude_dimensions: set[int] | None = None,
) -> None:
    if exclude_dimensions:
        rows = [
            row
            for row in rows
            if int(float(row["problem_dimension"])) not in exclude_dimensions
        ]

    plt.style.use(str(STYLE_PATH))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title("Training problem scaling -- Synthetic")

    by_solver: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_solver[row["solver"]].append(row)

    for solver in SOLVERS:
        solver_rows = by_solver.get(solver, [])
        if not solver_rows:
            continue

        color = SOLVER_COLORS[solver]
        light_color = _light_color(color)
        dims = np.array([float(r["problem_dimension"]) for r in solver_rows])
        times = np.array([float(r["training_time"]) for r in solver_rows])

        ax.scatter(
            dims,
            times,
            s=9,
            color=light_color,
            edgecolors=color,
            linewidths=0.3,
            alpha=0.55,
            zorder=3,
        )

        unique_dims = np.array(sorted(set(dims)))
        medians, q25s, q75s = [], [], []
        for dim in unique_dims:
            vals = times[dims == dim]
            medians.append(median(vals))
            q25s.append(np.percentile(vals, 25))
            q75s.append(np.percentile(vals, 75))
        med = np.array(medians)
        q25 = np.array(q25s)
        q75 = np.array(q75s)

        ax.fill_between(unique_dims, q25, q75, color=color, alpha=0.15, zorder=2)
        ax.plot(
            unique_dims,
            med,
            color=light_color,
            linewidth=2.0,
            label=SOLVER_LABELS[solver],
            zorder=6,
            path_effects=[
                path_effects.Stroke(linewidth=3.2, foreground=color),
                path_effects.Normal(),
            ],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Dimension of the training problem ($n \times m$)")
    ax.set_ylabel("Training time (s)")
    ax.grid(True, which="major", color="#cfcfcf", linestyle=":", linewidth=0.6)
    ax.grid(True, which="minor", color="#dedede", linestyle=":", linewidth=0.35)
    ax.tick_params(which="both", width=1.2)
    for spine in ax.spines.values():
        spine.set_color("black")

    ax.legend(loc="upper left")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-population", type=int, default=120000)
    parser.add_argument("--n-features", type=int, default=40)
    parser.add_argument("--n-min", type=int, default=100)
    parser.add_argument("--n-max", type=int, default=100000)
    parser.add_argument("--m-min", type=int, default=10)
    parser.add_argument("--m-max", type=int, default=1000)
    parser.add_argument("--grid-points", type=int, default=30)
    parser.add_argument("--n-grid", default="")
    parser.add_argument("--m-grid", default="")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--lam", type=float, default=1e-4)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--stages-per-restart", type=int, default=6)
    parser.add_argument("--m-inner0", type=int, default=512)
    parser.add_argument("--pegasos-epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--solvers", nargs="+", choices=SOLVERS, default=list(SOLVERS))
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument(
        "--append", action="store_true", help="Append only missing rows to the CSV."
    )
    parser.add_argument(
        "--exclude-dimensions",
        default=DEFAULT_EXCLUDE_DIMENSIONS,
        help="Comma-separated n*m dimensions to omit from the plot only.",
    )
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.n_population = 12000
        args.n_min = 1000
        args.n_max = 10000
        args.m_min = 40
        args.m_max = 120
        args.grid_points = 5
        args.repeats = 1
        args.restarts = 2
        args.stages_per_restart = 3
        args.pegasos_epochs = 5
    return args


def main() -> None:
    args = parse_args()
    if args.plot_only:
        rows = _read_rows(args.csv)
        print(f"Loaded {len(rows)} rows from {args.csv}")
    else:
        new_rows = run_training_problem_scaling(args)
        if args.append:
            _append_rows(args.csv, new_rows)
            rows = _read_rows(args.csv)
            print(
                f"Appended {len(new_rows)} rows to {args.csv}; loaded {len(rows)} total rows."
            )
        else:
            rows = new_rows
            _write_rows(args.csv, rows)
            print(f"Wrote {len(rows)} rows to {args.csv}")

    exclude_dimensions = (
        set(_parse_ints(args.exclude_dimensions)) if args.exclude_dimensions else set()
    )
    if exclude_dimensions:
        print(f"Plot-only excluded dimensions: {sorted(exclude_dimensions)}")
    plot_training_problem_scaling(rows, args.out, exclude_dimensions=exclude_dimensions)


if __name__ == "__main__":
    main()
