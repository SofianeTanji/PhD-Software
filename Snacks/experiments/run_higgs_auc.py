#!/usr/bin/env python3
"""Compute HIGGS 1 - AUC at validation-selected benchmark configs.

The canonical benchmark CSV stores accuracies, not decision scores, so AUC
cannot be computed retroactively. This script reruns only selected HIGGS
configurations and writes a compact metric CSV.

Usage:
    PYTHONPATH=src .venv/bin/python experiments/run_higgs_auc.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev as _stdev

import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC

from benchmark import transform_chunked
from lib.common import (
    RASSG_R_DEFAULT_BETA0_SCALE,
    RASSG_R_DEFAULT_BETA_DECAY,
    RASSG_R_DEFAULT_GROWTH,
    RASSG_R_DEFAULT_M_INNER0,
    RASSG_R_DEFAULT_RESTARTS,
    RASSG_R_DEFAULT_STAGES_PER_RESTART,
    Timer,
)
from lib.datasets import load_dataset
from run_diagnostics import _benchmark_selected_rassg_configs
from snacks.kernels import median_bandwidth, rbf_kernel
from snacks.nystrom import NystromTransformer, select_columns
from snacks.solver import rassg_r

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_RESULTS_CSV = RESULTS_DIR / "raw_results.csv"
DEFAULT_OUT = RESULTS_DIR / "higgs_auc.csv"
DEFAULT_SOLVERS = ("RASSG-r", "Pegasos", "LibLinear", "sklearn-Nystrom")
SOLVER_ALIASES = {"Snacks": "RASSG-r"}

FIELDS = [
    "dataset",
    "solver",
    "seed",
    "lam",
    "m",
    "raw_dtype",
    "feature_dtype",
    "estimated_peak_gib",
    "roc_auc",
    "one_minus_auc",
    "test_acc",
    "train_time_solver",
    "predict_time_solver",
    "notes",
]


def _std(xs: list[float]) -> float:
    return _stdev(xs) if len(xs) > 1 else 0.0


def _selected_higgs_config(config_csv: Path) -> tuple[float, int]:
    configs = _benchmark_selected_rassg_configs(config_csv)
    if "HIGGS" not in configs:
        raise ValueError(f"No validation-selected RASSG-r HIGGS config in {config_csv}")
    return configs["HIGGS"]


def _higgs_config(args: argparse.Namespace) -> tuple[float, int]:
    lam, m = _selected_higgs_config(Path(args.config_csv))
    if args.lam is not None:
        lam = args.lam
    if args.m is not None:
        m = args.m
    return lam, m


def _dtype_from_name(name: str) -> np.dtype:
    return np.dtype(np.float32)


def _chunk_rows(n_components: int, chunk_mb: float, itemsize: int) -> int | None:
    if chunk_mb <= 0:
        return None
    bytes_per_row = max(1, 3 * n_components * itemsize)
    return max(1, int(chunk_mb * 1024 * 1024 // bytes_per_row))


def _estimate_dense_peak_gib(
    split, m: int, transform_chunk_mb: float, raw_dtype, feature_dtype
) -> float:
    """Estimate peak memory after the script densifies LIBSVM matrices."""
    m_eff = min(m, split.n_train)
    n_train = split.X_train.shape[0]
    n_val = split.X_val.shape[0]
    n_test = split.X_test.shape[0]
    n_features = split.X_train.shape[1]
    raw_itemsize = np.dtype(raw_dtype).itemsize
    feature_itemsize = np.dtype(feature_dtype).itemsize
    dense_x = (n_train + n_val + n_test) * n_features * raw_itemsize
    z_train = n_train * m_eff * feature_itemsize
    z_val = n_val * m_eff * feature_itemsize
    z_test = n_test * m_eff * feature_itemsize
    eig_workspace = 3 * m_eff * m_eff * np.dtype(np.float32).itemsize
    rows_per_chunk = _chunk_rows(m_eff, transform_chunk_mb, feature_itemsize)
    if rows_per_chunk is None:
        chunk_tmp = max(z_train, z_val, z_test)
    else:
        chunk_tmp = (
            min(max(n_train, n_val, n_test), rows_per_chunk)
            * m_eff
            * feature_itemsize
            * 3
        )
    return (dense_x + z_train + z_val + z_test + eig_workspace + chunk_tmp) / (1024**3)


def _metric_row(
    *,
    solver: str,
    seed: int,
    lam: float,
    m: int,
    y_test: np.ndarray,
    scores: np.ndarray,
    raw_dtype: str,
    feature_dtype: str,
    estimated_peak_gib: float,
    train_time_solver: float | str = "",
    predict_time_solver: float | str = "",
    notes: str = "",
) -> dict:
    y_true = y_test > 0
    auc = float(roc_auc_score(y_true, scores))
    test_acc = float(np.mean((scores > 0) == y_true))
    return {
        "dataset": "HIGGS",
        "solver": solver,
        "seed": seed,
        "lam": lam,
        "m": m,
        "raw_dtype": raw_dtype,
        "feature_dtype": feature_dtype,
        "estimated_peak_gib": estimated_peak_gib,
        "roc_auc": auc,
        "one_minus_auc": 1.0 - auc,
        "test_acc": test_acc,
        "train_time_solver": train_time_solver,
        "predict_time_solver": predict_time_solver,
        "notes": notes,
    }


def run_seed(
    split,
    lam: float,
    m: int,
    seed: int,
    solvers: set[str],
    transform_chunk_mb: float,
    raw_dtype,
    feature_dtype,
    estimated_peak_gib: float,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    raw_dtype = np.dtype(raw_dtype)
    feature_dtype = np.dtype(feature_dtype)
    X_tr = np.asarray(
        split.X_train.toarray() if hasattr(split.X_train, "toarray") else split.X_train,
        dtype=raw_dtype,
    )
    X_val = np.asarray(
        split.X_val.toarray() if hasattr(split.X_val, "toarray") else split.X_val,
        dtype=raw_dtype,
    )
    X_te = np.asarray(
        split.X_test.toarray() if hasattr(split.X_test, "toarray") else split.X_test,
        dtype=raw_dtype,
    )
    y_tr, y_te = split.y_train, split.y_test
    n_tr = len(y_tr)

    gamma = float(
        median_bandwidth(X_tr, n_subsample=2000, rng=np.random.default_rng(seed))
    )
    kernel = lambda A, B: rbf_kernel(A, B, gamma)  # noqa: E731
    rows: list[dict] = []

    shared_solvers = solvers.intersection({"RASSG-r", "Pegasos", "LibLinear"})
    if shared_solvers:
        columns = select_columns(X_tr, min(m, n_tr), rng)
        nys = NystromTransformer(
            kernel,
            columns,
            ridge_mu=1e-6,
            output_dtype=feature_dtype,
        ).fit()
        Z_tr = transform_chunked(nys, X_tr, transform_chunk_mb)
        Z_val = transform_chunked(nys, X_val, transform_chunk_mb)
        Z_te = transform_chunked(nys, X_te, transform_chunk_mb)

    if "RASSG-r" in solvers:
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
        with Timer() as t_solver:
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
                y_val=split.y_val,
                rng=np.random.default_rng(seed),
            )
        with Timer() as t_pred:
            scores = Z_te @ res.u
        optimizer_time = (
            res.optimizer_time if res.optimizer_time is not None else t_solver.elapsed
        )
        rows.append(
            _metric_row(
                solver="RASSG-r",
                seed=seed,
                lam=lam,
                m=m,
                y_test=y_te,
                scores=scores,
                raw_dtype=raw_dtype.name,
                feature_dtype=feature_dtype.name,
                estimated_peak_gib=estimated_peak_gib,
                train_time_solver=optimizer_time,
                predict_time_solver=t_pred.elapsed,
                notes=(
                    "selected by RASSG-r validation accuracy; solver-only timing; "
                    f"validation_selection_time={res.selection_time if res.selection_time is not None else 0.0:.6g}"
                ),
            )
        )

    if "Pegasos" in solvers:
        clf = SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=lam,
            learning_rate="optimal",
            fit_intercept=False,
            max_iter=20,
            shuffle=True,
            random_state=seed,
        )
        with Timer() as t_solver:
            clf.fit(Z_tr, y_tr)
        with Timer() as t_pred:
            scores = clf.decision_function(Z_te)
        rows.append(
            _metric_row(
                solver="Pegasos",
                seed=seed,
                lam=lam,
                m=m,
                y_test=y_te,
                scores=scores,
                raw_dtype=raw_dtype.name,
                feature_dtype=feature_dtype.name,
                estimated_peak_gib=estimated_peak_gib,
                train_time_solver=t_solver.elapsed,
                predict_time_solver=t_pred.elapsed,
                notes=f"same HIGGS config as RASSG-r; input_feature_dtype={Z_tr.dtype.name}; coef_dtype={clf.coef_.dtype.name}",
            )
        )

    if "LibLinear" in solvers:
        clf = LinearSVC(
            loss="hinge",
            penalty="l2",
            C=1 / (n_tr * lam),
            fit_intercept=False,
            dual=True,
            max_iter=50000,
            random_state=seed,
        )
        with Timer() as t_solver:
            clf.fit(Z_tr, y_tr)
        with Timer() as t_pred:
            scores = clf.decision_function(Z_te)
        rows.append(
            _metric_row(
                solver="LibLinear",
                seed=seed,
                lam=lam,
                m=m,
                y_test=y_te,
                scores=scores,
                raw_dtype=raw_dtype.name,
                feature_dtype=feature_dtype.name,
                estimated_peak_gib=estimated_peak_gib,
                train_time_solver=t_solver.elapsed,
                predict_time_solver=t_pred.elapsed,
                notes=f"same HIGGS config as RASSG-r; input_feature_dtype={Z_tr.dtype.name}; coef_dtype={clf.coef_.dtype.name}",
            )
        )

    if "sklearn-Nystrom" in solvers:
        sk_nys = Nystroem(
            kernel="rbf", gamma=gamma, n_components=min(m, n_tr), random_state=seed
        )
        clf = LinearSVC(
            loss="hinge",
            penalty="l2",
            C=1 / (n_tr * lam),
            fit_intercept=False,
            dual=True,
            max_iter=50000,
            random_state=seed,
        )
        with Timer() as t_solver:
            sk_nys.fit(X_tr)
            Z_tr_sk = transform_chunked(
                sk_nys, X_tr, transform_chunk_mb, output_dtype=feature_dtype
            )
            clf.fit(Z_tr_sk, y_tr)
        with Timer() as t_pred:
            Z_te_sk = transform_chunked(
                sk_nys, X_te, transform_chunk_mb, output_dtype=feature_dtype
            )
            scores = clf.decision_function(Z_te_sk)
        rows.append(
            _metric_row(
                solver="sklearn-Nystrom",
                seed=seed,
                lam=lam,
                m=m,
                y_test=y_te,
                scores=scores,
                raw_dtype=raw_dtype.name,
                feature_dtype=feature_dtype.name,
                estimated_peak_gib=estimated_peak_gib,
                train_time_solver=t_solver.elapsed,
                predict_time_solver=t_pred.elapsed,
                notes=f"end-to-end embedding plus solver timing; coef_dtype={clf.coef_.dtype.name}",
            )
        )

    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config-csv", default=str(DEFAULT_RESULTS_CSV))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument(
        "--lam",
        type=float,
        default=None,
        help="Override lambda. Defaults to the validation-selected RASSG-r HIGGS config.",
    )
    p.add_argument(
        "--m",
        type=int,
        default=None,
        help="Override Nyström rank. Defaults to the validation-selected RASSG-r HIGGS config.",
    )
    p.add_argument(
        "--raw-dtype",
        choices=["float32"],
        default="float32",
        help="Dense dtype for raw HIGGS features.",
    )
    p.add_argument(
        "--feature-dtype",
        choices=["float32"],
        default="float32",
        help="Dense dtype for stored Nyström features.",
    )
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument(
        "--solvers",
        default=",".join(DEFAULT_SOLVERS),
        help=f"Comma-separated subset of {DEFAULT_SOLVERS}.",
    )
    p.add_argument("--transform-chunk-mb", type=float, default=512.0)
    p.add_argument(
        "--max-peak-gb",
        type=float,
        default=24.0,
        help="Refuse configs whose estimated dense peak exceeds this GiB budget. Use 0 to disable.",
    )
    args = p.parse_args()

    solvers = {
        SOLVER_ALIASES.get(s.strip(), s.strip())
        for s in args.solvers.split(",")
        if s.strip()
    }
    unknown = solvers.difference(DEFAULT_SOLVERS)
    if unknown:
        raise ValueError(f"Unknown solver(s): {', '.join(sorted(unknown))}")

    lam, m = _higgs_config(args)
    raw_dtype = _dtype_from_name(args.raw_dtype)
    feature_dtype = _dtype_from_name(args.feature_dtype)
    print(f"HIGGS selected config: lam={lam:g} m={m}")
    print(f"dtypes: raw={raw_dtype.name} features={feature_dtype.name}")
    split = load_dataset("HIGGS")
    peak_gib = _estimate_dense_peak_gib(
        split, m, args.transform_chunk_mb, raw_dtype, feature_dtype
    )
    print(f"Estimated dense peak memory: {peak_gib:.2f} GiB")
    if args.max_peak_gb > 0 and peak_gib > args.max_peak_gb:
        raise MemoryError(
            f"Estimated peak {peak_gib:.2f} GiB exceeds --max-peak-gb={args.max_peak_gb:.2f}. "
            "Lower --m or pass a larger/zero budget if this machine can handle it."
        )

    rows: list[dict] = []
    for seed in range(args.n_seeds):
        seed_rows = run_seed(
            split,
            lam,
            m,
            seed,
            solvers,
            args.transform_chunk_mb,
            raw_dtype,
            feature_dtype,
            peak_gib,
        )
        rows.extend(seed_rows)
        for row in seed_rows:
            print(
                f"  seed={seed} {row['solver']}: "
                f"1-AUC={row['one_minus_auc']:.6f} acc={row['test_acc']:.4f}"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    by_solver: dict[str, list[float]] = {}
    for row in rows:
        by_solver.setdefault(row["solver"], []).append(float(row["one_minus_auc"]))
    print(f"\nWrote {len(rows)} rows to {out}")
    for solver in sorted(by_solver):
        xs = by_solver[solver]
        print(f"{solver}: 1-AUC={mean(xs):.6f} +/- {_std(xs):.6f}")


if __name__ == "__main__":
    main()
