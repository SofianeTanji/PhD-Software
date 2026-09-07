#!/usr/bin/env python3
"""Plot L1-SVM training objective versus solver-only training time.

Synthetic data are generated directly in the embedded feature space. Real
datasets are embedded once with a shared Nyström map before the solvers run.
In both cases the x-axis is strictly solver training time: no kernel, Nyström
fit, or feature-transform time is included.

Usage:
    PYTHONPATH=src uv run python experiments/plot_l1_svm_loss.py --smoke
    PYTHONPATH=src uv run python experiments/plot_l1_svm_loss.py --n-train 200000 --m 300
    PYTHONPATH=src uv run python experiments/plot_l1_svm_loss.py --dataset SUSY --max-train 500000 --m 300
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numba
import numpy as np
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).parent.parent))
import experiments  # noqa: E402,F401  # registers BuRd colormap

from lib.datasets import load_dataset  # noqa: E402
from lib.plotstyle import apply_style, save_fig  # noqa: E402
from snacks.kernels import median_bandwidth, rbf_kernel  # noqa: E402
from snacks.nystrom import NystromTransformer, select_columns  # noqa: E402
from snacks.solver import _inner_stage_regularized_l1, rassg_r_l1  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
DEFAULT_CSV = RESULTS_DIR / "l1_svm_loss_trace.csv"
DEFAULT_FIGURE_STEM = "l1_svm_loss"
DEFAULT_TIME_BUDGET = 2.0
DEFAULT_SIGNAL_NORM = 8.0
DEFAULT_LABEL_NOISE = 0.10
DEFAULT_CONDITION_NUMBER = 10_000.0
DEFAULT_SNACKS_RESTARTS = 3_000
DEFAULT_SNACKS_STAGES_PER_RESTART = 5
DEFAULT_SNACKS_M_INNER0 = 256
DEFAULT_SNACKS_GROWTH = 1.0
DEFAULT_SNACKS_BETA0_SCALE = 80.0
DEFAULT_SNACKS_BETA_DECAY = 2.0
DEFAULT_SNACKS_RECORD_EVERY = 50
DEFAULT_REAL_SNACKS_RESTARTS = 20_000
DEFAULT_REAL_SNACKS_M_INNER0 = 192
DEFAULT_REAL_SNACKS_BETA0_SCALE = 0.5
DEFAULT_PEGASOS_ETA0 = 2.0
DEFAULT_PEGASOS_ETA_POWER = 0.5
PAPER_SNACKS_STAGES_PER_RESTART = 5
PAPER_SNACKS_GROWTH = 1.15
PAPER_PEGASOS_ETA_POWER = 0.5
DEFAULT_TRANSFORM_CHUNK_MB = 512.0
DEFAULT_MEDIAN_SUBSAMPLE = 2_000
TRANSFORM_CHUNK_TMP_FACTOR = 6

COLORS = {
    "Snacks": "#194F8C",
    "Pegasos": "#EB9486",
    "LibLinear": "#A0B78F",
}
FIELDS = [
    "solver",
    "dataset",
    "point_type",
    "step",
    "n_train",
    "m",
    "eval_size",
    "lam",
    "train_time_solver",
    "objective",
    "hinge_loss",
    "l1_term",
    "weight_l1_norm",
    "weight_l2_norm",
    "weight_nnz",
    "optimized_loss",
    "target_metric",
    "target_loss",
    "target_reached",
    "time_budget",
    "time_budget_reached",
    "notes",
]


def _dense(arr, dtype=np.float32) -> np.ndarray:
    return np.asarray(arr.toarray() if hasattr(arr, "toarray") else arr, dtype=dtype)


def _chunk_rows(n_components: int, chunk_mb: float, itemsize: int = 4) -> int | None:
    if chunk_mb <= 0:
        return None
    bytes_per_row = max(1, TRANSFORM_CHUNK_TMP_FACTOR * n_components * itemsize)
    return max(1, int(chunk_mb * 1024 * 1024 // bytes_per_row))


def _transform_chunked(
    transformer: NystromTransformer,
    X: np.ndarray,
    chunk_mb: float,
) -> np.ndarray:
    """Transform X in row chunks to cap temporary kernel memory."""
    n = X.shape[0]
    rows_per_chunk = _chunk_rows(
        transformer.columns.shape[0],
        chunk_mb,
        np.dtype(np.float32).itemsize,
    )
    if rows_per_chunk is None or rows_per_chunk >= n:
        return transformer.transform(X).astype(np.float32, copy=False)

    first_end = min(rows_per_chunk, n)
    first = transformer.transform(X[:first_end]).astype(np.float32, copy=False)
    Z = np.empty((n, first.shape[1]), dtype=np.float32)
    Z[:first_end] = first
    del first
    for start in range(first_end, n, rows_per_chunk):
        end = min(start + rows_per_chunk, n)
        Z[start:end] = transformer.transform(X[start:end]).astype(
            np.float32,
            copy=False,
        )
    return Z


def _synthetic_embedded_svm(
    n_train: int,
    m: int,
    seed: int,
    *,
    informative_fraction: float = 0.15,
    label_noise: float = DEFAULT_LABEL_NOISE,
    signal_norm: float = DEFAULT_SIGNAL_NORM,
    condition_number: float = DEFAULT_CONDITION_NUMBER,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    feature_scales = np.geomspace(1.0, 1.0 / np.sqrt(condition_number), m).astype(
        np.float32
    )
    Z = rng.normal(0.0, 1.0 / np.sqrt(m), size=(n_train, m)).astype(np.float32)
    Z *= feature_scales
    n_informative = max(1, int(round(m * informative_fraction)))
    w_true = np.zeros(m, dtype=np.float32)
    support = rng.choice(m, size=n_informative, replace=False)
    w_true[support] = rng.normal(0.0, 1.0, size=n_informative).astype(np.float32)
    clean_scores = Z @ w_true
    clean_scale = float(np.std(clean_scores))
    w_true *= np.float32(signal_norm / max(1e-6, clean_scale))
    scores = Z @ w_true + rng.normal(0.0, label_noise, size=n_train).astype(np.float32)
    y = np.where(scores >= 0.0, 1.0, -1.0).astype(np.float32)
    return Z, y


def _real_embedded_svm(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    split = load_dataset(args.dataset, seed=args.seed, dtype=np.float32)
    load_time = time.perf_counter() - t0

    n_available = split.n_train
    n_features = split.n_features
    is_sparse = split.is_sparse
    X_raw = split.X_train
    y_all = split.y_train
    del split
    gc.collect()

    if args.max_train > 0 and args.max_train < n_available:
        idx = rng.choice(n_available, size=args.max_train, replace=False)
        X_selected = X_raw[idx]
        y = np.ascontiguousarray(y_all[idx], dtype=np.float32)
    else:
        X_selected = X_raw
        y = np.ascontiguousarray(y_all, dtype=np.float32)
    del X_raw, y_all
    gc.collect()

    X = np.ascontiguousarray(_dense(X_selected, dtype=np.float32))
    del X_selected
    gc.collect()

    m_eff = min(args.m, X.shape[0])
    gamma = median_bandwidth(
        X,
        n_subsample=min(args.median_subsample, X.shape[0]),
        rng=rng,
    )
    columns = select_columns(X, m_eff, rng)

    def kernel(A: np.ndarray, B: np.ndarray, Y_sq: np.ndarray | None = None):
        return rbf_kernel(A, B, gamma, Y_sq=Y_sq)

    t_fit = time.perf_counter()
    nystrom = NystromTransformer(
        kernel,
        columns,
        ridge_mu=args.nystrom_ridge_mu,
        output_dtype=np.float32,
    ).fit()
    fit_time = time.perf_counter() - t_fit

    t_transform = time.perf_counter()
    Z = _transform_chunked(nystrom, X, args.transform_chunk_mb)
    transform_time = time.perf_counter() - t_transform
    del X, columns, nystrom
    gc.collect()

    info = {
        "load_time": load_time,
        "fit_time": fit_time,
        "transform_time": transform_time,
        "gamma": gamma,
        "n_available": n_available,
        "n_features": n_features,
        "is_sparse": is_sparse,
    }
    return Z, y, info


def _objective(Z: np.ndarray, y: np.ndarray, u: np.ndarray, lam: float) -> dict:
    scores = Z @ u
    margins = y * scores
    hinge = np.maximum(0.0, 1.0 - margins)
    l1_norm = float(np.sum(np.abs(u)))
    l1_term = lam * l1_norm
    return {
        "objective": float(np.mean(hinge)) + l1_term,
        "hinge_loss": float(np.mean(hinge)),
        "l1_term": l1_term,
        "weight_l1_norm": l1_norm,
        "weight_l2_norm": float(np.sqrt(np.dot(u, u))),
        "weight_nnz": int(np.count_nonzero(np.abs(u) > 1e-8)),
    }


@numba.njit(cache=True)
def _pegasos_l1_subgradient_chunk(
    w: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    t_start: int,
    lam: float,
    eta0: float,
    eta_power: float,
) -> np.ndarray:
    """Run explicit stochastic subgradient steps for hinge + lambda ||w||_1."""
    r = w.shape[0]
    for k in range(indices.shape[0]):
        i = indices[k]
        t = t_start + k + 1
        eta = eta0 / (np.float32(t) ** eta_power)

        d = np.float32(0.0)
        for j in range(r):
            d += w[j] * Z[i, j]
        active = y[i] * d < np.float32(1.0)

        reg_coef = eta * lam
        if active:
            hinge_coef = eta * y[i]
            for j in range(r):
                if w[j] > np.float32(0.0):
                    reg_subgrad = np.float32(1.0)
                elif w[j] < np.float32(0.0):
                    reg_subgrad = np.float32(-1.0)
                else:
                    reg_subgrad = np.float32(0.0)
                w[j] = w[j] - reg_coef * reg_subgrad + hinge_coef * Z[i, j]
        else:
            for j in range(r):
                if w[j] > np.float32(0.0):
                    reg_subgrad = np.float32(1.0)
                elif w[j] < np.float32(0.0):
                    reg_subgrad = np.float32(-1.0)
                else:
                    reg_subgrad = np.float32(0.0)
                w[j] = w[j] - reg_coef * reg_subgrad
    return w


def _row(
    *,
    solver: str,
    dataset: str,
    point_type: str,
    step: int,
    n_train: int,
    m: int,
    eval_size: int,
    lam: float,
    train_time_solver: float,
    metrics: dict,
    optimized_loss: str,
    target_metric: str,
    target_loss: float | None,
    time_budget: float | None,
    notes: str = "",
) -> dict:
    target_reached = (
        target_loss is not None and float(metrics[target_metric]) <= target_loss
    )
    time_budget_reached = (
        time_budget is not None and train_time_solver >= time_budget
    )
    return {
        "solver": solver,
        "dataset": dataset,
        "point_type": point_type,
        "step": step,
        "n_train": n_train,
        "m": m,
        "eval_size": eval_size,
        "lam": lam,
        "train_time_solver": train_time_solver,
        "optimized_loss": optimized_loss,
        "target_metric": target_metric,
        "target_loss": "" if target_loss is None else target_loss,
        "target_reached": target_reached,
        "time_budget": "" if time_budget is None else time_budget,
        "time_budget_reached": time_budget_reached,
        "notes": notes,
        **metrics,
    }


def _target_reached(metrics: dict, args: argparse.Namespace) -> bool:
    return args.target_loss is not None and metrics[args.y_metric] <= args.target_loss


def _time_budget_reached(elapsed: float, args: argparse.Namespace) -> bool:
    return args.time_budget is not None and elapsed >= args.time_budget


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _snacks_trace(
    Z: np.ndarray,
    y: np.ndarray,
    Z_eval: np.ndarray,
    y_eval: np.ndarray,
    args: argparse.Namespace,
) -> list[dict]:
    # Warm up compilation outside the measured run.
    rassg_r_l1(
        Z[:2],
        y[:2],
        lam=args.lam,
        restarts=1,
        stages_per_restart=1,
        m_inner0=1,
        growth=1.0,
        rng=np.random.default_rng(0),
    )

    rng = np.random.default_rng(args.seed)
    lam = np.float32(args.lam)
    n_train, m = Z.shape
    center = np.zeros(m, dtype=np.float32)
    rows = [
        _row(
            solver="Snacks",
            dataset=args.dataset,
            point_type="initial",
            step=0,
            n_train=n_train,
            m=m,
            eval_size=len(y_eval),
            lam=args.lam,
            train_time_solver=0.0,
            metrics=(metrics := _objective(Z_eval, y_eval, center, args.lam)),
            optimized_loss="hinge",
            target_metric=args.y_metric,
            target_loss=args.target_loss,
            time_budget=args.time_budget,
        )
    ]
    if _target_reached(metrics, args):
        return rows
    cumulative = 0.0
    global_stage = 0

    for restart in range(1, args.snacks_restarts + 1):
        m_inner = max(1, int(round(args.snacks_m_inner0 * (args.snacks_growth ** (restart - 1)))))
        beta = args.snacks_beta0_scale / (lam * np.sqrt(np.float32(m_inner)))
        suffix_start = min(int(np.floor((1.0 - args.alpha) * m_inner)), m_inner - 1)

        for local_stage in range(1, args.snacks_stages_per_restart + 1):
            global_stage += 1
            indices = rng.integers(0, n_train, size=m_inner)
            acc_mask = np.zeros(m_inner, dtype=np.bool_)
            acc_mask[rng.integers(suffix_start, m_inner, size=args.q)] = True

            t0 = time.perf_counter()
            center = _inner_stage_regularized_l1(
                center,
                Z,
                y,
                indices,
                lam,
                np.float32(beta),
                acc_mask,
            )
            cumulative += time.perf_counter() - t0
            budget_stop = _time_budget_reached(cumulative, args) and (
                local_stage == args.snacks_stages_per_restart
            )
            should_record = (
                global_stage % args.snacks_record_every == 0
                or budget_stop
                or args.target_loss is not None
            )
            if should_record:
                metrics = _objective(Z_eval, y_eval, center, args.lam)
                rows.append(
                    _row(
                        solver="Snacks",
                        dataset=args.dataset,
                        point_type="stage",
                        step=global_stage,
                        n_train=n_train,
                        m=m,
                        eval_size=len(y_eval),
                        lam=args.lam,
                        train_time_solver=cumulative,
                        metrics=metrics,
                        optimized_loss="hinge",
                        target_metric=args.y_metric,
                        target_loss=args.target_loss,
                        time_budget=args.time_budget,
                        notes=(
                            f"restart={restart}; local_stage={local_stage}; "
                            f"m_inner={m_inner}"
                        ),
                    )
                )
                if _target_reached(metrics, args):
                    return rows
            if budget_stop:
                return rows
            beta /= args.snacks_beta_decay

    return rows


def _pegasos_trace(
    Z: np.ndarray,
    y: np.ndarray,
    Z_eval: np.ndarray,
    y_eval: np.ndarray,
    args: argparse.Namespace,
) -> list[dict]:
    # Warm up compilation outside the measured run.
    _pegasos_l1_subgradient_chunk(
        np.zeros(Z.shape[1], dtype=np.float32),
        Z[:2],
        y[:2],
        np.arange(2, dtype=np.int64),
        0,
        np.float32(args.lam),
        np.float32(args.pegasos_eta0),
        np.float32(args.pegasos_eta_power),
    )

    n_train, m = Z.shape
    rng = np.random.default_rng(args.seed)
    w = np.zeros(m, dtype=np.float32)
    rows = [
        _row(
            solver="Pegasos",
            dataset=args.dataset,
            point_type="initial",
            step=0,
            n_train=n_train,
            m=m,
            eval_size=len(y_eval),
            lam=args.lam,
            train_time_solver=0.0,
            metrics=(metrics := _objective(Z_eval, y_eval, w, args.lam)),
            optimized_loss="hinge",
            target_metric=args.y_metric,
            target_loss=args.target_loss,
            time_budget=args.time_budget,
        )
    ]
    if _target_reached(metrics, args):
        return rows
    cumulative = 0.0
    batch_size = min(args.pegasos_batch_size, n_train)
    max_chunks = int(np.ceil(args.pegasos_epochs * n_train / batch_size))
    samples_seen = 0
    for step in range(1, max_chunks + 1):
        indices = rng.integers(0, n_train, size=batch_size, dtype=np.int64)
        t0 = time.perf_counter()
        w = _pegasos_l1_subgradient_chunk(
            w,
            Z,
            y,
            indices,
            samples_seen,
            np.float32(args.lam),
            np.float32(args.pegasos_eta0),
            np.float32(args.pegasos_eta_power),
        )
        cumulative += time.perf_counter() - t0
        samples_seen += batch_size
        metrics = _objective(Z_eval, y_eval, w, args.lam)
        rows.append(
            _row(
                solver="Pegasos",
                dataset=args.dataset,
                point_type="batch",
                step=step,
                n_train=n_train,
                m=m,
                eval_size=len(y_eval),
                lam=args.lam,
                train_time_solver=cumulative,
                metrics=metrics,
                optimized_loss="hinge",
                target_metric=args.y_metric,
                target_loss=args.target_loss,
                time_budget=args.time_budget,
                notes=(
                    f"samples_seen={samples_seen}; batch_size={batch_size}; "
                    f"eta0={args.pegasos_eta0:g}; eta_power={args.pegasos_eta_power:g}; "
                    "regularizer_subgradient=sign(w)"
                ),
            )
        )
        if _target_reached(metrics, args) or _time_budget_reached(cumulative, args):
            return rows
    return rows


def _liblinear_point(
    Z: np.ndarray,
    y: np.ndarray,
    Z_eval: np.ndarray,
    y_eval: np.ndarray,
    args: argparse.Namespace,
) -> list[dict]:
    n_train, m = Z.shape
    clf = LinearSVC(
        loss="squared_hinge",
        penalty="l1",
        C=1 / (n_train * args.lam),
        fit_intercept=False,
        dual=False,
        max_iter=args.liblinear_max_iter,
        random_state=args.seed,
    )
    t0 = time.perf_counter()
    clf.fit(Z, y)
    elapsed = time.perf_counter() - t0
    u = clf.coef_.ravel().astype(np.float32, copy=False)
    return [
        _row(
            solver="LibLinear",
            dataset=args.dataset,
            point_type="final",
            step=1,
            n_train=n_train,
            m=m,
            eval_size=len(y_eval),
            lam=args.lam,
            train_time_solver=elapsed,
            metrics=_objective(Z_eval, y_eval, u, args.lam),
            optimized_loss="squared_hinge",
            target_metric=args.y_metric,
            target_loss=args.target_loss,
            time_budget=args.time_budget,
            notes="LinearSVC penalty=l1 uses squared_hinge; plotted against hinge+L1 objective",
        )
    ]


def _plot(
    rows: list[dict],
    figure_dir: Path,
    stem: str,
    title: str,
    y_metric: str,
    x_scale: str,
    y_scale: str,
    plot_best_so_far: bool,
    gap_to_best: bool,
    gap_eps: float,
) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    y_label = {
        "objective": r"Training objective: hinge + $\lambda\|w\|_1$",
        "hinge_loss": "Training hinge loss",
    }[y_metric]
    if plot_best_so_far:
        y_label = "Best " + y_label[0].lower() + y_label[1:] + " so far"
    reference = None
    if gap_to_best:
        reference = min(
            float(row[y_metric])
            for row in rows
            if row["solver"] in {"Snacks", "Pegasos"}
        )
        y_label = {
            "objective": "Best objective gap",
            "hinge_loss": "Best hinge-loss gap",
        }[y_metric] if plot_best_so_far else {
            "objective": "Objective gap",
            "hinge_loss": "Hinge-loss gap",
        }[y_metric]
    positive_times = [
        float(row["train_time_solver"])
        for row in rows
        if row["solver"] in {"Snacks", "Pegasos"}
        and row["point_type"] != "initial"
        and float(row["train_time_solver"]) > 0
    ]
    initial_plot_time = min(positive_times) / 2.0 if positive_times else 1e-6

    for solver in ("Snacks", "Pegasos"):
        points = [row for row in rows if row["solver"] == solver]
        if not points:
            continue
        x = [
            initial_plot_time
            if row["point_type"] == "initial" and x_scale == "log"
            else float(row["train_time_solver"])
            for row in points
        ]
        y = [float(row[y_metric]) for row in points]
        if plot_best_so_far:
            y = np.minimum.accumulate(np.asarray(y, dtype=np.float64)).tolist()
        if reference is not None:
            y = np.maximum(np.asarray(y, dtype=np.float64) - reference, gap_eps).tolist()
        ax.plot(
            x,
            y,
            marker="o",
            markersize=3.0,
            linewidth=1.25,
            color=COLORS[solver],
            label=solver,
        )

    liblinear = [row for row in rows if row["solver"] == "LibLinear"]
    if liblinear:
        row = liblinear[-1]
        y_value = float(row[y_metric])
        if reference is not None:
            y_value = max(y_value - reference, gap_eps)
        ax.scatter(
            [float(row["train_time_solver"])],
            [y_value],
            s=42,
            marker="D",
            color=COLORS["LibLinear"],
            edgecolor="black",
            linewidth=0.5,
            label="LibLinear (final)",
            zorder=5,
        )

    target_rows = [
        row
        for row in rows
        if row.get("target_metric") == y_metric and row.get("target_loss") != ""
    ]
    if target_rows:
        target_loss = float(target_rows[0]["target_loss"])
        ax.axhline(
            target_loss,
            color="#555555",
            linestyle="--",
            linewidth=0.9,
            label=f"target {target_loss:g}",
        )

    budget_rows = [row for row in rows if row.get("time_budget") != ""]
    if budget_rows:
        time_budget = float(budget_rows[0]["time_budget"])
        ax.axvline(
            time_budget,
            color="#555555",
            linestyle=":",
            linewidth=0.9,
            label=f"{time_budget:g}s budget",
        )

    ax.set_xlabel("Solver training time (s)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    ax.legend(loc="best")
    ax.margins(x=0.04, y=0.08)
    save_fig(fig, figure_dir, stem, png=True)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n-train", type=int, default=200_000)
    parser.add_argument(
        "--dataset",
        default="synthetic",
        help="Use 'synthetic' or any dataset registered in experiments/lib/datasets.py.",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=0,
        help="For real datasets, randomly cap the training rows. Use 0 for all rows.",
    )
    parser.add_argument("--m", type=int, default=300)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-size", type=int, default=50_000)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--figure-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--figure-stem", default=DEFAULT_FIGURE_STEM)
    parser.add_argument(
        "--y-metric", choices=["objective", "hinge_loss"], default="hinge_loss"
    )
    parser.add_argument("--x-scale", choices=["linear", "log"], default="log")
    parser.add_argument("--y-scale", choices=["linear", "log"], default="log")
    parser.add_argument(
        "--raw-trace",
        action="store_true",
        help="Plot raw stochastic iterates instead of best-so-far loss.",
    )
    parser.add_argument(
        "--gap-to-best",
        action="store_true",
        help="Plot loss minus the best observed traced Snacks/Pegasos loss.",
    )
    parser.add_argument("--gap-eps", type=float, default=1e-5)
    parser.add_argument(
        "--time-budget",
        type=float,
        default=DEFAULT_TIME_BUDGET,
        help="Run each traced solver until cumulative solver time reaches this many seconds. "
        "Use a negative value to disable time-budget stopping.",
    )
    parser.add_argument(
        "--target-loss",
        type=float,
        default=-1.0,
        help="Stop each traced solver once the plotted metric reaches this value. "
        "Use a negative value to disable target stopping.",
    )
    parser.add_argument("--signal-norm", type=float, default=DEFAULT_SIGNAL_NORM)
    parser.add_argument("--label-noise", type=float, default=DEFAULT_LABEL_NOISE)
    parser.add_argument(
        "--condition-number",
        type=float,
        default=DEFAULT_CONDITION_NUMBER,
        help="Geometric feature-covariance condition number for the synthetic problem.",
    )
    parser.add_argument(
        "--transform-chunk-mb",
        type=float,
        default=DEFAULT_TRANSFORM_CHUNK_MB,
        help="Temporary-memory target for real-dataset Nyström transforms.",
    )
    parser.add_argument(
        "--median-subsample",
        type=int,
        default=DEFAULT_MEDIAN_SUBSAMPLE,
        help="Number of real-dataset rows used by the RBF median bandwidth heuristic.",
    )
    parser.add_argument("--nystrom-ridge-mu", type=float, default=1e-6)
    parser.add_argument("--snacks-restarts", type=int, default=None)
    parser.add_argument(
        "--snacks-stages-per-restart",
        type=int,
        default=DEFAULT_SNACKS_STAGES_PER_RESTART,
    )
    parser.add_argument("--snacks-m-inner0", type=int, default=None)
    parser.add_argument("--snacks-growth", type=float, default=DEFAULT_SNACKS_GROWTH)
    parser.add_argument(
        "--snacks-beta0-scale", type=float, default=None
    )
    parser.add_argument(
        "--snacks-beta-decay", type=float, default=DEFAULT_SNACKS_BETA_DECAY
    )
    parser.add_argument("--snacks-record-every", type=int, default=DEFAULT_SNACKS_RECORD_EVERY)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--q", type=int, default=16)
    parser.add_argument("--pegasos-epochs", type=int, default=100)
    parser.add_argument("--pegasos-batch-size", type=int, default=2_000)
    parser.add_argument("--pegasos-eta0", type=float, default=DEFAULT_PEGASOS_ETA0)
    parser.add_argument(
        "--pegasos-eta-power",
        type=float,
        default=DEFAULT_PEGASOS_ETA_POWER,
        help="Explicit L1 subgradient schedule eta_t = eta0 / t**eta_power.",
    )
    parser.add_argument("--liblinear-max-iter", type=int, default=50_000)
    parser.add_argument("--skip-liblinear", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--paper-schedule",
        action="store_true",
        help=(
            "Use the Xu et al. schedule choices: eta_t=eta0/sqrt(t), "
            "restart every 5 stages, grow the inner horizon by 1.15 at each "
            "restart, and set Snacks beta from eta0."
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.snacks_restarts is None:
        args.snacks_restarts = (
            DEFAULT_SNACKS_RESTARTS
            if args.dataset == "synthetic"
            else DEFAULT_REAL_SNACKS_RESTARTS
        )
    if args.snacks_m_inner0 is None:
        args.snacks_m_inner0 = (
            DEFAULT_SNACKS_M_INNER0
            if args.dataset == "synthetic"
            else DEFAULT_REAL_SNACKS_M_INNER0
        )
    if args.snacks_beta0_scale is None:
        args.snacks_beta0_scale = (
            DEFAULT_SNACKS_BETA0_SCALE
            if args.dataset == "synthetic"
            else DEFAULT_REAL_SNACKS_BETA0_SCALE
        )
    if args.smoke:
        args.n_train = 5_000
        if args.dataset != "synthetic" and args.max_train == 0:
            args.max_train = args.n_train
        args.m = 80
        args.eval_size = 2_000
        args.snacks_restarts = 1
        args.snacks_stages_per_restart = 20
        args.snacks_m_inner0 = 64
        args.snacks_record_every = 1
        args.pegasos_epochs = 20
        args.pegasos_batch_size = 500
        args.time_budget = 0.05
        args.liblinear_max_iter = 5_000
        args.figure_stem = args.figure_stem + "_smoke"

    if args.paper_schedule:
        args.pegasos_eta_power = PAPER_PEGASOS_ETA_POWER
        args.snacks_stages_per_restart = PAPER_SNACKS_STAGES_PER_RESTART
        args.snacks_growth = PAPER_SNACKS_GROWTH
        args.snacks_beta0_scale = (
            args.pegasos_eta0 * args.lam * np.sqrt(np.float32(args.snacks_m_inner0))
        ) / 2.0

    if args.n_train <= 0:
        raise ValueError("--n-train must be positive.")
    if args.max_train < 0:
        raise ValueError("--max-train must be nonnegative.")
    if args.m <= 0:
        raise ValueError("--m must be positive.")
    if args.lam <= 0:
        raise ValueError("--lam must be positive.")
    if args.eval_size <= 0:
        raise ValueError("--eval-size must be positive.")
    if args.target_loss is not None and args.target_loss < 0:
        args.target_loss = None
    if args.time_budget is not None and args.time_budget < 0:
        args.time_budget = None
    if args.signal_norm <= 0:
        raise ValueError("--signal-norm must be positive.")
    if args.label_noise < 0:
        raise ValueError("--label-noise must be nonnegative.")
    if args.condition_number < 1:
        raise ValueError("--condition-number must be at least 1.")
    if args.transform_chunk_mb <= 0:
        raise ValueError("--transform-chunk-mb must be positive.")
    if args.median_subsample <= 1:
        raise ValueError("--median-subsample must be greater than 1.")
    if args.nystrom_ridge_mu <= 0:
        raise ValueError("--nystrom-ridge-mu must be positive.")
    if args.snacks_restarts <= 0 or args.snacks_stages_per_restart <= 0:
        raise ValueError("--snacks-restarts and --snacks-stages-per-restart must be positive.")
    if args.snacks_record_every <= 0:
        raise ValueError("--snacks-record-every must be positive.")
    if args.pegasos_epochs <= 0:
        raise ValueError("--pegasos-epochs must be positive.")
    if args.pegasos_batch_size <= 0:
        raise ValueError("--pegasos-batch-size must be positive.")
    if args.pegasos_eta0 <= 0:
        raise ValueError("--pegasos-eta0 must be positive.")
    if args.pegasos_eta_power < 0:
        raise ValueError("--pegasos-eta-power must be nonnegative.")
    if args.gap_eps <= 0:
        raise ValueError("--gap-eps must be positive.")

    if args.dataset != "synthetic" and args.figure_stem == DEFAULT_FIGURE_STEM:
        args.figure_stem = f"{DEFAULT_FIGURE_STEM}_{args.dataset.lower()}"
    if args.dataset != "synthetic" and args.csv == DEFAULT_CSV:
        args.csv = RESULTS_DIR / f"l1_svm_loss_trace_{args.dataset.lower()}.csv"

    if args.plot_only:
        rows = _read_csv(args.csv)
        if not rows:
            raise ValueError(f"No rows found in {args.csv}")
        dataset = rows[0].get("dataset", args.dataset)
        plot_title = (
            "L1-SVM on ill-conditioned synthetic data"
            if dataset == "synthetic"
            else f"L1-SVM on {dataset}"
        )
        _plot(
            rows,
            args.figure_dir,
            args.figure_stem,
            plot_title,
            args.y_metric,
            args.x_scale,
            args.y_scale,
            not args.raw_trace,
            args.gap_to_best,
            args.gap_eps,
        )
        print(f"Wrote figure to {args.figure_dir / (args.figure_stem + '.pdf')}")
        print(f"Wrote figure to {args.figure_dir / (args.figure_stem + '.png')}")
        return

    if args.dataset == "synthetic":
        print(
            f"Generating synthetic data: n_train={args.n_train}, "
            f"m={args.m}, lambda={args.lam:g}, signal_norm={args.signal_norm:g}, "
            f"label_noise={args.label_noise:g}, "
            f"condition_number={args.condition_number:g}, "
            f"time_budget={args.time_budget}"
        )
        Z, y = _synthetic_embedded_svm(
            args.n_train,
            args.m,
            args.seed,
            signal_norm=args.signal_norm,
            label_noise=args.label_noise,
            condition_number=args.condition_number,
        )
        plot_title = "L1-SVM on ill-conditioned synthetic data"
    else:
        cap = "all rows" if args.max_train == 0 else f"at most {args.max_train} rows"
        print(
            f"Loading {args.dataset}: training cap={cap}, m={args.m}, "
            f"lambda={args.lam:g}, time_budget={args.time_budget}"
        )
        Z, y, info = _real_embedded_svm(args)
        print(
            f"Prepared {args.dataset}: n_available={info['n_available']}, "
            f"n_used={len(y)}, raw_features={info['n_features']}, "
            f"sparse={info['is_sparse']}, gamma={info['gamma']:.4g}, "
            f"load={info['load_time']:.2f}s, nystrom_fit={info['fit_time']:.2f}s, "
            f"nystrom_transform={info['transform_time']:.2f}s "
            "(excluded from plotted solver time)"
        )
        plot_title = f"L1-SVM on {args.dataset}"

    args.n_train = len(y)
    args.m = Z.shape[1]
    eval_size = min(args.eval_size, args.n_train)
    rng = np.random.default_rng(args.seed + 1)
    eval_idx = rng.choice(args.n_train, size=eval_size, replace=False)
    Z_eval = np.ascontiguousarray(Z[eval_idx])
    y_eval = np.ascontiguousarray(y[eval_idx])

    rows: list[dict] = []
    print("Running Snacks trace")
    rows.extend(_snacks_trace(Z, y, Z_eval, y_eval, args))
    gc.collect()
    print("Running Pegasos trace")
    rows.extend(_pegasos_trace(Z, y, Z_eval, y_eval, args))
    gc.collect()
    if not args.skip_liblinear:
        print("Running LibLinear final point")
        rows.extend(_liblinear_point(Z, y, Z_eval, y_eval, args))

    _write_csv(args.csv, rows)
    _plot(
        rows,
        args.figure_dir,
        args.figure_stem,
        plot_title,
        args.y_metric,
        args.x_scale,
        args.y_scale,
        not args.raw_trace,
        args.gap_to_best,
        args.gap_eps,
    )

    print(f"Wrote trace CSV to {args.csv}")
    print(f"Wrote figure to {args.figure_dir / (args.figure_stem + '.pdf')}")
    print(f"Wrote figure to {args.figure_dir / (args.figure_stem + '.png')}")
    for solver in ("Snacks", "Pegasos", "LibLinear"):
        solver_rows = [row for row in rows if row["solver"] == solver]
        if not solver_rows:
            continue
        final = solver_rows[-1]
        best = min(solver_rows, key=lambda row: float(row[args.y_metric]))
        print(
            f"  {solver:<9} time={float(final['train_time_solver']):.3f}s "
            f"{args.y_metric}={float(final[args.y_metric]):.4g} "
            f"best_{args.y_metric}={float(best[args.y_metric]):.4g}"
            f"@{float(best['train_time_solver']):.3f}s "
            f"objective={float(final['objective']):.4g} "
            f"time_budget_reached={final['time_budget_reached']}"
        )


if __name__ == "__main__":
    main()
