#!/usr/bin/env python3
"""Minimal L1-regularized embedded SVM comparison.

Runs Snacks/RASSG-r, Pegasos/SGD, and LibLinear on the same Nyström features.
Snacks and Pegasos optimize mean hinge loss plus ``lambda * ||u||_1``.
sklearn's LibLinear wrapper does not support ``penalty="l1", loss="hinge"``,
so the LibLinear row uses its supported L1-regularized squared-hinge SVM and is
marked as a near-baseline in the output.

Usage:
    PYTHONPATH=src uv run python experiments/run_l1_svm.py --smoke
    PYTHONPATH=src uv run python experiments/run_l1_svm.py --dataset mushrooms --m 200
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

from benchmark import dense, densify_split, make_synthetic, transform_chunked
from lib.common import (
    RASSG_R_DEFAULT_BETA0_SCALE,
    RASSG_R_DEFAULT_BETA_DECAY,
    RASSG_R_DEFAULT_GROWTH,
    RASSG_R_DEFAULT_M_INNER0,
    RASSG_R_DEFAULT_RESTARTS,
    RASSG_R_DEFAULT_STAGES_PER_RESTART,
    Timer,
    make_run_id,
    save_metadata,
)
from lib.datasets import load_dataset
from snacks.kernels import median_bandwidth, rbf_kernel
from snacks.nystrom import NystromTransformer, select_columns
from snacks.solver import rassg_r_l1

SCRIPT_NAME = "run_l1_svm.py"
DEFAULT_OUT = Path("experiments/results/l1_svm.csv")
SOLVERS = ("Snacks", "Pegasos", "LibLinear")
SOLVER_ALIASES = {"RASSG-r": "Snacks", "RASSG-L1": "Snacks"}

FIELDS = [
    "run_id",
    "script_name",
    "dataset",
    "solver",
    "seed",
    "lam",
    "m",
    "gamma",
    "n_train",
    "n_val",
    "n_test",
    "n_features",
    "optimized_loss",
    "regularizer",
    "train_acc",
    "val_acc",
    "test_acc",
    "train_hinge_loss",
    "train_squared_hinge_loss",
    "train_l1_term",
    "train_hinge_l1_objective",
    "train_squared_hinge_l1_objective",
    "weight_l1_norm",
    "weight_l2_norm",
    "weight_nnz_tol",
    "weight_zero_fraction_tol",
    "zero_tol",
    "train_time_solver",
    "predict_time_solver",
    "nystrom_fit_time",
    "nystrom_transform_train_time",
    "nystrom_transform_val_time",
    "nystrom_transform_test_time",
    "total_train_time",
    "total_predict_time",
    "total_time",
    "n_stages_run",
    "stop_reason",
    "notes",
]


def _parse_seeds(value: str) -> list[int]:
    seeds = []
    for part in value.split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("--seeds must contain at least one integer.")
    return seeds


def _normalize_solvers(solvers: Iterable[str]) -> list[str]:
    selected = {SOLVER_ALIASES.get(solver, solver) for solver in solvers}
    unknown = selected.difference(SOLVERS)
    if unknown:
        raise ValueError(f"Unknown solver(s): {', '.join(sorted(unknown))}")
    return [solver for solver in SOLVERS if solver in selected]


def _append_rows(out_path: Path, rows: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDS})


def _acc(Z: np.ndarray, y: np.ndarray, u: np.ndarray) -> float:
    return float(np.mean(((Z @ u) > 0) == (y > 0)))


def _weight_stats(u: np.ndarray, tol: float) -> dict:
    abs_u = np.abs(u)
    nnz = int(np.count_nonzero(abs_u > tol))
    return {
        "weight_l1_norm": float(np.sum(abs_u)),
        "weight_l2_norm": float(np.sqrt(np.dot(u, u))),
        "weight_nnz_tol": nnz,
        "weight_zero_fraction_tol": float(1.0 - nnz / max(1, u.size)),
        "zero_tol": tol,
    }


def _objective_stats(Z: np.ndarray, y: np.ndarray, u: np.ndarray, lam: float) -> dict:
    margins = y * (Z @ u)
    hinge = np.maximum(0.0, 1.0 - margins)
    squared_hinge = hinge * hinge
    l1_term = lam * float(np.sum(np.abs(u)))
    return {
        "train_hinge_loss": float(np.mean(hinge)),
        "train_squared_hinge_loss": float(np.mean(squared_hinge)),
        "train_l1_term": l1_term,
        "train_hinge_l1_objective": float(np.mean(hinge)) + l1_term,
        "train_squared_hinge_l1_objective": float(np.mean(squared_hinge)) + l1_term,
    }


def _make_row(
    *,
    base: dict,
    solver: str,
    optimized_loss: str,
    u: np.ndarray,
    Z_tr: np.ndarray,
    y_tr: np.ndarray,
    Z_val: np.ndarray,
    y_val: np.ndarray,
    Z_te: np.ndarray,
    y_te: np.ndarray,
    lam: float,
    train_time_solver: float,
    predict_time_solver: float,
    nystrom_fit_time: float,
    nystrom_transform_train_time: float,
    nystrom_transform_val_time: float,
    nystrom_transform_test_time: float,
    n_stages_run: int | str = "",
    stop_reason: str = "",
    notes: str = "",
    zero_tol: float = 1e-8,
) -> dict:
    row = {
        **base,
        "solver": solver,
        "optimized_loss": optimized_loss,
        "regularizer": "l1",
        "train_acc": _acc(Z_tr, y_tr, u),
        "val_acc": _acc(Z_val, y_val, u),
        "test_acc": _acc(Z_te, y_te, u),
        "train_time_solver": train_time_solver,
        "predict_time_solver": predict_time_solver,
        "nystrom_fit_time": nystrom_fit_time,
        "nystrom_transform_train_time": nystrom_transform_train_time,
        "nystrom_transform_val_time": nystrom_transform_val_time,
        "nystrom_transform_test_time": nystrom_transform_test_time,
        "total_train_time": (
            nystrom_fit_time + nystrom_transform_train_time + train_time_solver
        ),
        "total_predict_time": nystrom_transform_test_time + predict_time_solver,
        "total_time": (
            nystrom_fit_time
            + nystrom_transform_train_time
            + nystrom_transform_val_time
            + nystrom_transform_test_time
            + train_time_solver
            + predict_time_solver
        ),
        "n_stages_run": n_stages_run,
        "stop_reason": stop_reason,
        "notes": notes,
    }
    row.update(_objective_stats(Z_tr, y_tr, u, lam))
    row.update(_weight_stats(u, zero_tol))
    return row


def run_config(
    *,
    split,
    dataset: str,
    lam: float,
    m: int,
    seed: int,
    run_id: str,
    solvers: Iterable[str],
    transform_chunk_mb: float,
    snacks_restarts: int,
    snacks_stages_per_restart: int,
    snacks_m_inner0: int,
    snacks_growth: float,
    snacks_beta0_scale: float,
    snacks_beta_decay: float,
    pegasos_epochs: int,
    liblinear_max_iter: int,
    zero_tol: float,
) -> list[dict]:
    selected_solvers = _normalize_solvers(solvers)
    rng = np.random.default_rng(seed)
    X_tr = dense(split.X_train, dtype=np.float32)
    X_val = dense(split.X_val, dtype=np.float32)
    X_te = dense(split.X_test, dtype=np.float32)
    y_tr, y_val, y_te = split.y_train, split.y_val, split.y_test
    n_tr = len(y_tr)

    gamma = float(
        median_bandwidth(X_tr, n_subsample=2000, rng=np.random.default_rng(seed))
    )
    kernel = lambda A, B: rbf_kernel(A, B, gamma)  # noqa: E731

    columns = select_columns(X_tr, min(m, n_tr), rng)
    with Timer() as t_fit:
        nys = NystromTransformer(
            kernel,
            columns,
            ridge_mu=1e-6,
            output_dtype=np.float32,
        ).fit()
    with Timer() as t_tr_z:
        Z_tr = transform_chunked(nys, X_tr, transform_chunk_mb)
    with Timer() as t_val_z:
        Z_val = transform_chunked(nys, X_val, transform_chunk_mb)
    with Timer() as t_te_z:
        Z_te = transform_chunked(nys, X_te, transform_chunk_mb)

    base = {
        "run_id": run_id,
        "script_name": SCRIPT_NAME,
        "dataset": dataset,
        "seed": seed,
        "lam": lam,
        "m": min(m, n_tr),
        "gamma": gamma,
        "n_train": n_tr,
        "n_val": len(y_val),
        "n_test": len(y_te),
        "n_features": split.n_features,
    }
    transform_times = {
        "nystrom_fit_time": t_fit.elapsed,
        "nystrom_transform_train_time": t_tr_z.elapsed,
        "nystrom_transform_val_time": t_val_z.elapsed,
        "nystrom_transform_test_time": t_te_z.elapsed,
    }
    rows: list[dict] = []

    if "Snacks" in selected_solvers:
        # Warm up numba compilation outside the measured solver call.
        rassg_r_l1(
            Z_tr[:2],
            y_tr[:2],
            lam=lam,
            restarts=1,
            stages_per_restart=1,
            m_inner0=1,
            growth=1.0,
            rng=np.random.default_rng(0),
        )
        with Timer() as t_solver:
            res = rassg_r_l1(
                Z_tr,
                y_tr,
                lam=lam,
                restarts=snacks_restarts,
                stages_per_restart=snacks_stages_per_restart,
                m_inner0=snacks_m_inner0,
                growth=snacks_growth,
                beta0_scale=snacks_beta0_scale,
                beta_decay=snacks_beta_decay,
                Z_val=Z_val,
                y_val=y_val,
                rng=np.random.default_rng(seed),
            )
        with Timer() as t_pred:
            _ = Z_te @ res.u
        optimizer_time = (
            res.optimizer_time if res.optimizer_time is not None else t_solver.elapsed
        )
        selection_time = (
            res.selection_time
            if res.selection_time is not None
            else max(0.0, t_solver.elapsed - optimizer_time)
        )
        rows.append(
            _make_row(
                base=base,
                solver="Snacks",
                optimized_loss="hinge",
                u=res.u,
                Z_tr=Z_tr,
                y_tr=y_tr,
                Z_val=Z_val,
                y_val=y_val,
                Z_te=Z_te,
                y_te=y_te,
                lam=lam,
                train_time_solver=optimizer_time,
                predict_time_solver=t_pred.elapsed,
                **transform_times,
                n_stages_run=res.n_stages_run,
                stop_reason=res.stop_reason,
                notes=(
                    "RASSG-r with L1 oracle sign(u); "
                    f"wall_solver_time={t_solver.elapsed:.6g}; "
                    f"validation_selection_time={selection_time:.6g}; "
                    f"restarts={snacks_restarts}; "
                    f"stages_per_restart={snacks_stages_per_restart}; "
                    f"m_inner0={snacks_m_inner0}; growth={snacks_growth:g}; "
                    f"beta0_scale={snacks_beta0_scale:g}; "
                    f"beta_decay={snacks_beta_decay:g}"
                ),
                zero_tol=zero_tol,
            )
        )

    if "Pegasos" in selected_solvers:
        clf = SGDClassifier(
            loss="hinge",
            penalty="l1",
            alpha=lam,
            learning_rate="optimal",
            fit_intercept=False,
            max_iter=pegasos_epochs,
            tol=None,
            shuffle=True,
            random_state=seed,
        )
        with Timer() as t_solver:
            clf.fit(Z_tr, y_tr)
        u = clf.coef_.ravel().astype(np.float32, copy=False)
        with Timer() as t_pred:
            _ = clf.predict(Z_te)
        rows.append(
            _make_row(
                base=base,
                solver="Pegasos",
                optimized_loss="hinge",
                u=u,
                Z_tr=Z_tr,
                y_tr=y_tr,
                Z_val=Z_val,
                y_val=y_val,
                Z_te=Z_te,
                y_te=y_te,
                lam=lam,
                train_time_solver=t_solver.elapsed,
                predict_time_solver=t_pred.elapsed,
                **transform_times,
                n_stages_run=pegasos_epochs,
                stop_reason="max_iter",
                notes=(
                    "SGDClassifier hinge penalty=l1; "
                    f"epochs={pegasos_epochs}; input_feature_dtype={Z_tr.dtype.name}; "
                    f"coef_dtype={clf.coef_.dtype.name}"
                ),
                zero_tol=zero_tol,
            )
        )

    if "LibLinear" in selected_solvers:
        clf = LinearSVC(
            loss="squared_hinge",
            penalty="l1",
            C=1 / (n_tr * lam),
            fit_intercept=False,
            dual=False,
            max_iter=liblinear_max_iter,
            random_state=seed,
        )
        with Timer() as t_solver:
            clf.fit(Z_tr, y_tr)
        u = clf.coef_.ravel().astype(np.float32, copy=False)
        with Timer() as t_pred:
            _ = clf.predict(Z_te)
        rows.append(
            _make_row(
                base=base,
                solver="LibLinear",
                optimized_loss="squared_hinge",
                u=u,
                Z_tr=Z_tr,
                y_tr=y_tr,
                Z_val=Z_val,
                y_val=y_val,
                Z_te=Z_te,
                y_te=y_te,
                lam=lam,
                train_time_solver=t_solver.elapsed,
                predict_time_solver=t_pred.elapsed,
                **transform_times,
                n_stages_run="",
                stop_reason="",
                notes=(
                    "LinearSVC penalty=l1 requires loss=squared_hinge in sklearn; "
                    "not the exact hinge+L1 objective; "
                    f"C={1 / (n_tr * lam):.6g}; input_feature_dtype={Z_tr.dtype.name}; "
                    f"coef_dtype={clf.coef_.dtype.name}"
                ),
                zero_tol=zero_tol,
            )
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--m", type=int, default=50)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--solvers",
        default=",".join(SOLVERS),
        help=f"Comma-separated subset of {SOLVERS}.",
    )
    parser.add_argument("--transform-chunk-mb", type=float, default=512.0)
    parser.add_argument("--snacks-restarts", type=int, default=RASSG_R_DEFAULT_RESTARTS)
    parser.add_argument(
        "--snacks-stages-per-restart",
        type=int,
        default=RASSG_R_DEFAULT_STAGES_PER_RESTART,
    )
    parser.add_argument("--snacks-m-inner0", type=int, default=RASSG_R_DEFAULT_M_INNER0)
    parser.add_argument("--snacks-growth", type=float, default=RASSG_R_DEFAULT_GROWTH)
    parser.add_argument(
        "--snacks-beta0-scale", type=float, default=RASSG_R_DEFAULT_BETA0_SCALE
    )
    parser.add_argument(
        "--snacks-beta-decay", type=float, default=RASSG_R_DEFAULT_BETA_DECAY
    )
    parser.add_argument("--pegasos-epochs", type=int, default=20)
    parser.add_argument("--liblinear-max-iter", type=int, default=50000)
    parser.add_argument("--zero-tol", type=float, default=1e-8)
    args = parser.parse_args()

    if args.lam <= 0:
        raise ValueError("--lam must be positive.")
    if args.m <= 0:
        raise ValueError("--m must be positive.")
    if args.zero_tol < 0:
        raise ValueError("--zero-tol must be nonnegative.")

    if args.smoke:
        args.seeds = "0"
        args.pegasos_epochs = min(args.pegasos_epochs, 5)
        args.snacks_restarts = 1
        args.snacks_stages_per_restart = min(args.snacks_stages_per_restart, 5)
        args.snacks_m_inner0 = min(args.snacks_m_inner0, 128)

    seeds = _parse_seeds(args.seeds)
    solvers = _normalize_solvers(
        solver.strip() for solver in args.solvers.split(",") if solver.strip()
    )
    split = (
        make_synthetic(seed=0)
        if args.dataset == "synthetic"
        else load_dataset(args.dataset, dtype=np.float32)
    )
    split = densify_split(split, dtype=np.float32)
    run_id = make_run_id()
    out_path = Path(args.out)

    total_rows = 0
    print(
        f"dataset={args.dataset} n_train={split.n_train} m={min(args.m, split.n_train)} "
        f"lambda={args.lam:g} solvers={','.join(solvers)}"
    )
    for seed in seeds:
        rows = run_config(
            split=split,
            dataset=args.dataset,
            lam=args.lam,
            m=args.m,
            seed=seed,
            run_id=run_id,
            solvers=solvers,
            transform_chunk_mb=args.transform_chunk_mb,
            snacks_restarts=args.snacks_restarts,
            snacks_stages_per_restart=args.snacks_stages_per_restart,
            snacks_m_inner0=args.snacks_m_inner0,
            snacks_growth=args.snacks_growth,
            snacks_beta0_scale=args.snacks_beta0_scale,
            snacks_beta_decay=args.snacks_beta_decay,
            pegasos_epochs=args.pegasos_epochs,
            liblinear_max_iter=args.liblinear_max_iter,
            zero_tol=args.zero_tol,
        )
        _append_rows(out_path, rows)
        total_rows += len(rows)
        for row in rows:
            print(
                f"  seed={seed} {row['solver']:<9} "
                f"loss={row['optimized_loss']:<13} "
                f"test_acc={row['test_acc']:.4f} "
                f"hinge+l1={row['train_hinge_l1_objective']:.4g} "
                f"zeros={100 * row['weight_zero_fraction_tol']:.1f}% "
                f"solver={row['train_time_solver']:.3f}s"
            )

    save_metadata(out_path.parent, run_id, [sys.executable] + sys.argv)
    print(f"Wrote {total_rows} rows to {out_path}")


if __name__ == "__main__":
    main()
