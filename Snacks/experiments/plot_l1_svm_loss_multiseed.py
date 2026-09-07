#!/usr/bin/env python3
"""Run the paper-schedule L1-SVM synthetic comparison across multiple seeds.

The single tuned parameter is ``eta0``. Pegasos uses
``eta_t = eta0 / sqrt(t)``. Snacks uses the paper schedule and sets its
first-stage beta from the same ``eta0``:

    beta0_scale = eta0 * lambda * sqrt(T0) / 2.

The CSV stores raw traces for every seed. The figure interpolates best-so-far
loss onto a common log-time grid and plots mean +/- one standard deviation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import experiments  # noqa: E402,F401  # registers BuRd colormap

import plot_l1_svm_loss as single  # noqa: E402
from lib.plotstyle import apply_style, save_fig  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
DEFAULT_CSV = RESULTS_DIR / "l1_svm_loss_trace_synthetic_paper_30s_seeds.csv"
DEFAULT_FIGURE_STEM = "l1_svm_loss_synthetic_paper_30s_seeds"
FIELDS = ["seed", *single.FIELDS]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _build_single_args(args: argparse.Namespace, seed: int) -> argparse.Namespace:
    out = argparse.Namespace(
        dataset="synthetic",
        lam=args.lam,
        seed=seed,
        y_metric=args.y_metric,
        target_loss=None,
        time_budget=args.time_budget,
        snacks_restarts=args.snacks_restarts,
        snacks_stages_per_restart=single.PAPER_SNACKS_STAGES_PER_RESTART,
        snacks_m_inner0=args.snacks_m_inner0,
        snacks_growth=single.PAPER_SNACKS_GROWTH,
        snacks_beta0_scale=0.0,
        snacks_beta_decay=args.snacks_beta_decay,
        snacks_record_every=args.snacks_record_every,
        alpha=args.alpha,
        q=args.q,
        pegasos_epochs=args.pegasos_epochs,
        pegasos_batch_size=args.pegasos_batch_size,
        pegasos_eta0=args.eta0,
        pegasos_eta_power=single.PAPER_PEGASOS_ETA_POWER,
    )
    out.snacks_beta0_scale = (
        args.eta0 * args.lam * np.sqrt(np.float32(args.snacks_m_inner0))
    ) / 2.0
    return out


def _run_seed(args: argparse.Namespace, seed: int) -> list[dict]:
    single_args = _build_single_args(args, seed)
    print(
        f"[seed {seed}] generating synthetic: n={args.n_train}, m={args.m}, "
        f"eta0={args.eta0:g}, beta0_scale={single_args.snacks_beta0_scale:.4g}",
        flush=True,
    )
    Z, y = single._synthetic_embedded_svm(
        args.n_train,
        args.m,
        seed,
        signal_norm=args.signal_norm,
        label_noise=args.label_noise,
        condition_number=args.condition_number,
    )
    eval_size = min(args.eval_size, args.n_train)
    rng = np.random.default_rng(seed + 1)
    eval_idx = rng.choice(args.n_train, size=eval_size, replace=False)
    Z_eval = np.ascontiguousarray(Z[eval_idx])
    y_eval = np.ascontiguousarray(y[eval_idx])

    print(f"[seed {seed}] running Snacks", flush=True)
    rows = single._snacks_trace(Z, y, Z_eval, y_eval, single_args)
    gc.collect()
    print(f"[seed {seed}] running Pegasos", flush=True)
    rows.extend(single._pegasos_trace(Z, y, Z_eval, y_eval, single_args))
    for row in rows:
        row["seed"] = seed
    for solver in ("Snacks", "Pegasos"):
        solver_rows = [row for row in rows if row["solver"] == solver]
        best = min(solver_rows, key=lambda row: float(row[args.y_metric]))
        print(
            f"[seed {seed}] {solver:<7} best_{args.y_metric}="
            f"{float(best[args.y_metric]):.4g}@"
            f"{float(best['train_time_solver']):.3f}s",
            flush=True,
        )
    del Z, y, Z_eval, y_eval
    gc.collect()
    return rows


def _best_so_far_trace(rows: list[dict], y_metric: str) -> tuple[np.ndarray, np.ndarray]:
    positive_times = [
        float(row["train_time_solver"])
        for row in rows
        if row["point_type"] != "initial" and float(row["train_time_solver"]) > 0.0
    ]
    initial_time = min(positive_times) / 2.0 if positive_times else 1e-6
    x = np.asarray(
        [
            initial_time
            if row["point_type"] == "initial"
            else float(row["train_time_solver"])
            for row in rows
        ],
        dtype=np.float64,
    )
    y = np.asarray([float(row[y_metric]) for row in rows], dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    y = np.minimum.accumulate(y[order])
    keep = np.r_[True, np.diff(x) > 0.0]
    return x[keep], y[keep]


def _interp_best_trace(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    idx = np.searchsorted(x, grid, side="right") - 1
    idx = np.clip(idx, 0, len(y) - 1)
    return y[idx]


def _plot_aggregate(rows: list[dict], args: argparse.Namespace) -> None:
    apply_style()
    solvers = ("Snacks", "Pegasos")
    traces: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {solver: [] for solver in solvers}
    for solver in solvers:
        solver_rows = [row for row in rows if row["solver"] == solver]
        seeds = sorted({int(row["seed"]) for row in solver_rows})
        for seed in seeds:
            seed_rows = [row for row in solver_rows if int(row["seed"]) == seed]
            traces[solver].append(_best_so_far_trace(seed_rows, args.y_metric))

    all_positive = [
        x_val
        for solver_traces in traces.values()
        for x, _ in solver_traces
        for x_val in x
        if x_val > 0.0
    ]
    if not all_positive:
        raise ValueError("No positive solver times found.")
    x_min = max(min(all_positive), 1e-8)
    x_max = min(
        args.time_budget,
        max(max(x) for solver_traces in traces.values() for x, _ in solver_traces),
    )
    grid = np.geomspace(x_min, x_max, args.grid_size)

    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    for solver in solvers:
        values = np.vstack(
            [_interp_best_trace(x, y, grid) for x, y in traces[solver]]
        )
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1) if values.shape[0] > 1 else np.zeros_like(mean)
        ax.plot(
            grid,
            mean,
            color=single.COLORS[solver],
            linewidth=1.5,
            label=solver,
        )
        ax.fill_between(
            grid,
            np.maximum(mean - std, args.y_floor),
            mean + std,
            color=single.COLORS[solver],
            alpha=0.22,
            linewidth=0.0,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Solver training time (s)")
    ax.set_ylabel("Best loss so far")
    ax.set_title("L1-SVM (Synthetic)")
    ax.legend(loc="best")
    ax.margins(x=0.04, y=0.08)
    save_fig(fig, args.figure_dir, args.figure_stem, png=False)
    plt.close(fig)


def _parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
        if not seeds:
            raise ValueError("--seeds did not contain any seeds.")
        return seeds
    return list(range(args.seed_start, args.seed_start + args.n_seeds))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--n-train", type=int, default=300_000)
    parser.add_argument("--m", type=int, default=800)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--eta0", type=float, default=30_000.0)
    parser.add_argument("--eval-size", type=int, default=30_000)
    parser.add_argument("--signal-norm", type=float, default=single.DEFAULT_SIGNAL_NORM)
    parser.add_argument("--label-noise", type=float, default=0.2)
    parser.add_argument("--condition-number", type=float, default=1_000_000.0)
    parser.add_argument("--time-budget", type=float, default=30.0)
    parser.add_argument("--snacks-restarts", type=int, default=3_000)
    parser.add_argument("--snacks-m-inner0", type=int, default=single.DEFAULT_SNACKS_M_INNER0)
    parser.add_argument("--snacks-beta-decay", type=float, default=single.DEFAULT_SNACKS_BETA_DECAY)
    parser.add_argument("--snacks-record-every", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--q", type=int, default=16)
    parser.add_argument("--pegasos-epochs", type=int, default=2_000)
    parser.add_argument("--pegasos-batch-size", type=int, default=2_000)
    parser.add_argument(
        "--y-metric", choices=["objective", "hinge_loss"], default="hinge_loss"
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--figure-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--figure-stem", default=DEFAULT_FIGURE_STEM)
    parser.add_argument("--grid-size", type=int, default=300)
    parser.add_argument("--y-floor", type=float, default=1e-8)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive.")
    if args.n_train <= 0 or args.m <= 0:
        raise ValueError("--n-train and --m must be positive.")
    if args.lam <= 0 or args.eta0 <= 0:
        raise ValueError("--lam and --eta0 must be positive.")
    if args.time_budget <= 0:
        raise ValueError("--time-budget must be positive.")
    if args.grid_size < 2:
        raise ValueError("--grid-size must be at least 2.")
    if args.y_floor <= 0:
        raise ValueError("--y-floor must be positive.")

    seeds = _parse_seeds(args)
    args.n_seeds = len(seeds)
    rows = _read_csv(args.csv) if (args.reuse_existing or args.plot_only) else []
    done = {
        int(row["seed"])
        for row in rows
        if row.get("solver") == "Pegasos" and row.get("seed", "") != ""
    }
    if args.plot_only:
        missing = sorted(set(seeds) - done)
        if missing:
            raise ValueError(f"Missing seeds in {args.csv}: {missing}")
    else:
        for seed in seeds:
            if seed in done:
                print(f"[seed {seed}] already present in {args.csv}; skipping", flush=True)
                continue
            rows.extend(_run_seed(args, seed))
            _write_csv(args.csv, rows)
            print(f"[seed {seed}] appended rows to {args.csv}", flush=True)

    selected = set(seeds)
    plot_rows = [
        row for row in rows if row.get("seed", "") != "" and int(row["seed"]) in selected
    ]
    if not plot_rows:
        raise ValueError("No rows available for the selected seeds.")
    _plot_aggregate(plot_rows, args)
    print(f"Wrote trace CSV to {args.csv}")
    print(f"Wrote figure to {args.figure_dir / (args.figure_stem + '.pdf')}")


if __name__ == "__main__":
    main()
