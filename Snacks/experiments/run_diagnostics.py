#!/usr/bin/env python3
"""Run RASSG-r with per-stage diagnostics on representative datasets.

Saves per-stage history (objective, hinge loss, active fraction, timings, etc.)
to a CSV for convergence analysis.

Usage:
    PYTHONPATH=src uv run python experiments/run_diagnostics.py
    PYTHONPATH=src uv run python experiments/run_diagnostics.py --dataset ijcnn1
    PYTHONPATH=src uv run python experiments/run_diagnostics.py --out experiments/results/diagnostics.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

from lib.common import (
    RASSG_R_DEFAULT_BETA0_SCALE,
    RASSG_R_DEFAULT_BETA_DECAY,
    RASSG_R_DEFAULT_GROWTH,
    RASSG_R_DEFAULT_M_INNER0,
    RASSG_R_DEFAULT_RESTARTS,
    RASSG_R_DEFAULT_STAGES_PER_RESTART,
)
from lib.datasets import Split, load_dataset

from snacks.kernels import median_bandwidth, rbf_kernel
from snacks.nystrom import NystromTransformer, select_columns
from snacks.solver import rassg_r

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_RESULTS_CSV = RESULTS_DIR / "raw_results.csv"

DIAG_DATASETS = [
    "HIGGS",
    "YearPredictionMSD",
    "SUSY",
    "covtype.binary",
    "ijcnn1",
    "w8a",
]
DEFAULT_M = 200
DEFAULT_LAM = 1e-3
DATASET_M_OVERRIDES = {"mnist": 500}

DIAG_FIELDS = [
    "dataset",
    "solver",
    "seed",
    "lam",
    "m",
    "n_train",
    "stage",
    "restart",
    "local_stage",
    "m_inner",
    "grad_evals",
    "epoch",
    "eta",
    "beta",
    "stage_time",
    "train_acc",
    "val_acc",
    "test_acc",
    "train_objective",
    "train_hinge_loss",
    "regularization_term",
    "active_hinge_fraction",
    "weight_norm",
]


def make_synthetic(seed: int = 0) -> Split:
    X, y = make_classification(n_samples=2000, n_features=20, random_state=seed)
    X = X.astype(np.float32, copy=False)
    y = np.where(y == 0, -1.0, 1.0).astype(np.float32)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=seed)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=seed
    )
    return Split(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_val,
        y_val=y_val,
        X_test=X_te,
        y_test=y_te,
        n_train=len(X_tr),
        n_features=X_tr.shape[1],
        is_sparse=False,
    )


def dense(arr) -> np.ndarray:
    return np.asarray(
        arr.toarray() if hasattr(arr, "toarray") else arr, dtype=np.float32
    )


def _f(s) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _benchmark_selected_rassg_configs(path: Path) -> dict[str, tuple[float, int]]:
    """Return the validation-selected RASSG-r (lambda, m) config per dataset."""
    if not path.exists():
        return {}

    by_config: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("solver") != "RASSG-r":
                continue
            val_acc = _f(row.get("val_acc"))
            if val_acc is None:
                continue
            by_config[(row["dataset"], row.get("lam", ""), row.get("m", ""))].append(
                val_acc
            )

    best: dict[str, tuple[float, int, float]] = {}
    for (dataset, lam_s, m_s), vals in by_config.items():
        lam = _f(lam_s)
        m = _f(m_s)
        if lam is None or m is None:
            continue
        val_acc = mean(vals)
        if dataset not in best or val_acc > best[dataset][2]:
            best[dataset] = (lam, int(m), val_acc)

    return {dataset: (lam, m) for dataset, (lam, m, _) in best.items()}


def _diagnostic_config(
    dataset: str,
    args: argparse.Namespace,
    benchmark_configs: dict[str, tuple[float, int]],
) -> tuple[float, int]:
    bench_lam, bench_m = benchmark_configs.get(dataset, (DEFAULT_LAM, DEFAULT_M))
    lam = args.lam if args.lam is not None else bench_lam
    m = args.m if args.m is not None else bench_m
    if args.m is None and dataset not in benchmark_configs:
        m = DATASET_M_OVERRIDES.get(dataset, DEFAULT_M)
    return lam, m


def _subsample_rows(X, y, n: int | None, rng: np.random.Generator):
    if n is None or len(y) <= n:
        return X, y
    idx = rng.choice(len(y), size=n, replace=False)
    return X[idx], y[idx]


def _limit_diagnostic_split(
    split: Split,
    *,
    max_train: int | None,
    max_eval: int | None,
    seed: int,
) -> Split:
    if max_train is None and max_eval is None:
        return split

    rng = np.random.default_rng(seed)
    X_train, y_train = _subsample_rows(split.X_train, split.y_train, max_train, rng)
    X_val, y_val = _subsample_rows(split.X_val, split.y_val, max_eval, rng)
    X_test, y_test = _subsample_rows(split.X_test, split.y_test, max_eval, rng)
    return Split(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        n_train=split.n_train,
        n_features=split.n_features,
        is_sparse=split.is_sparse,
    )


def run_rassg_diagnostics_dataset(
    split: Split,
    dataset: str,
    lam: float = 1e-3,
    m: int = 200,
    seed: int = 0,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    X_tr = dense(split.X_train)
    X_val = dense(split.X_val)
    X_te = dense(split.X_test)
    n_tr = len(split.y_train)
    n_train_label = split.n_train

    gamma = float(
        median_bandwidth(X_tr, n_subsample=2000, rng=np.random.default_rng(seed))
    )
    columns = select_columns(X_tr, min(m, n_tr), rng)
    nys = NystromTransformer(
        lambda A, B: rbf_kernel(A, B, gamma), columns, ridge_mu=1e-6
    ).fit()
    Z_tr = nys.transform(X_tr)
    Z_val = nys.transform(X_val)
    Z_te = nys.transform(X_te)

    rassg_r(
        Z_tr[:2],
        split.y_train[:2],
        lam=lam,
        restarts=1,
        stages_per_restart=1,
        m_inner0=1,
        growth=1.0,
        rng=np.random.default_rng(0),
    )

    res = rassg_r(
        Z_tr,
        split.y_train,
        lam=lam,
        restarts=RASSG_R_DEFAULT_RESTARTS,
        stages_per_restart=RASSG_R_DEFAULT_STAGES_PER_RESTART,
        m_inner0=RASSG_R_DEFAULT_M_INNER0,
        growth=RASSG_R_DEFAULT_GROWTH,
        beta0_scale=RASSG_R_DEFAULT_BETA0_SCALE,
        beta_decay=RASSG_R_DEFAULT_BETA_DECAY,
        Z_val=Z_val,
        y_val=split.y_val,
        Z_test=Z_te,
        y_test=split.y_test,
        rng=np.random.default_rng(seed),
        compute_diagnostics=True,
    )

    rows = []
    grad_evals = 0
    for h in res.history:
        m_inner_stage = int(h.get("m_inner", RASSG_R_DEFAULT_M_INNER0))
        grad_evals += m_inner_stage
        rows.append(
            {
                "dataset": dataset,
                "solver": "RASSG-r",
                "seed": seed,
                "lam": lam,
                "m": m,
                "n_train": n_train_label,
                "stage": h["stage"],
                "restart": h.get("restart", ""),
                "local_stage": h.get("local_stage", ""),
                "m_inner": m_inner_stage,
                "grad_evals": grad_evals,
                "epoch": grad_evals / n_tr,
                "eta": h.get("eta", ""),
                "beta": h.get("beta", ""),
                "stage_time": h.get("stage_time", ""),
                "train_acc": h.get("train_acc", ""),
                "val_acc": h.get("val_acc", ""),
                "test_acc": h.get("test_acc", ""),
                "train_objective": h.get("train_objective", ""),
                "train_hinge_loss": h.get("train_hinge_loss", ""),
                "regularization_term": h.get("regularization_term", ""),
                "active_hinge_fraction": h.get("active_hinge_fraction", ""),
                "weight_norm": h.get("weight_norm", ""),
            }
        )
    return rows


def run_pegasos_diagnostics_dataset(
    split: Split,
    dataset: str,
    lam: float = 1e-3,
    m: int = 200,
    seed: int = 0,
    n_stages: int = 20,
    max_grad_evals: int | None = None,
) -> list[dict]:
    rng_seed = np.random.default_rng(seed)
    X_tr = dense(split.X_train)
    X_val = dense(split.X_val)
    X_te = dense(split.X_test)
    n_tr = len(split.y_train)
    n_train_label = split.n_train

    gamma = float(
        median_bandwidth(X_tr, n_subsample=2000, rng=np.random.default_rng(seed))
    )
    columns = select_columns(X_tr, min(m, n_tr), rng_seed)
    nys = NystromTransformer(
        lambda A, B: rbf_kernel(A, B, gamma), columns, ridge_mu=1e-6
    ).fit()
    Z_tr = nys.transform(X_tr)
    Z_val = nys.transform(X_val)
    Z_te = nys.transform(X_te)

    classes = np.array([-1.0, 1.0])
    clf = SGDClassifier(
        loss="hinge",
        penalty="l2",
        alpha=lam,
        learning_rate="optimal",
        fit_intercept=False,
        shuffle=False,
        random_state=seed,
    )

    # Check Pegasos at the first RASSG-r inner budget; max_grad_evals aligns
    # the final budget with the RASSG-r trajectory.
    chunk_size = RASSG_R_DEFAULT_M_INNER0
    rng = np.random.default_rng(seed)

    val_patience = 3
    min_stages = 5
    best_val_acc = -np.inf
    stages_no_improve = 0
    checkpoint = 0  # global chunk index across all epochs

    rows = []
    for epoch in range(1, n_stages + 1):
        perm = rng.permutation(n_tr)
        Z_tr_shuf = Z_tr[perm]
        y_tr_shuf = split.y_train[perm]

        start = 0
        while start < n_tr:
            grad_evals_before = (epoch - 1) * n_tr + start
            if max_grad_evals is not None and grad_evals_before >= max_grad_evals:
                return rows

            remaining_budget = (
                max_grad_evals - grad_evals_before
                if max_grad_evals is not None
                else chunk_size
            )
            end = min(start + chunk_size, n_tr, start + remaining_budget)
            t0 = time.perf_counter()
            clf.partial_fit(Z_tr_shuf[start:end], y_tr_shuf[start:end], classes=classes)
            stage_time = time.perf_counter() - t0
            checkpoint += 1
            grad_evals = (epoch - 1) * n_tr + end
            rows.append(
                {
                    "dataset": dataset,
                    "solver": "Pegasos",
                    "seed": seed,
                    "lam": lam,
                    "m": m,
                    "n_train": n_train_label,
                    "stage": checkpoint,
                    "restart": "",
                    "local_stage": "",
                    "m_inner": "",
                    "grad_evals": grad_evals,
                    "epoch": grad_evals / n_tr,
                    "eta": "",
                    "beta": "",
                    "stage_time": stage_time,
                    "train_acc": float(np.mean(clf.predict(Z_tr) == split.y_train)),
                    "val_acc": float(np.mean(clf.predict(Z_val) == split.y_val)),
                    "test_acc": float(np.mean(clf.predict(Z_te) == split.y_test)),
                    "train_objective": "",
                    "train_hinge_loss": "",
                    "regularization_term": "",
                    "active_hinge_fraction": "",
                    "weight_norm": "",
                }
            )
            start = end

        # Early stopping checked at epoch boundaries.
        if not rows:
            break
        val_acc = rows[-1]["val_acc"]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            stages_no_improve = 0
        else:
            stages_no_improve += 1

        if epoch >= min_stages and stages_no_improve >= val_patience:
            break

    return rows


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", default="all", help=f"'all' or one of {DIAG_DATASETS}")
    p.add_argument(
        "--lam",
        type=float,
        default=None,
        help="Override lambda. Defaults to the validation-selected RASSG-r benchmark config.",
    )
    p.add_argument(
        "--m",
        type=int,
        default=None,
        help="Override Nyström rank. Defaults to the validation-selected RASSG-r benchmark config.",
    )
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument(
        "--n-stages",
        type=int,
        default=20,
        help="Maximum Pegasos epochs; RASSG-r uses the shared restart default.",
    )
    p.add_argument(
        "--max-diagnostic-train",
        type=int,
        default=None,
        help="Optional cap on training rows used for diagnostics only.",
    )
    p.add_argument(
        "--max-diagnostic-eval",
        type=int,
        default=None,
        help="Optional cap on validation/test rows used for per-stage diagnostics only.",
    )
    p.add_argument(
        "--config-csv",
        default=str(DEFAULT_RESULTS_CSV),
        help="Benchmark CSV used to select RASSG-r diagnostic configs by validation accuracy.",
    )
    p.add_argument("--out", default="experiments/results/diagnostics.csv")
    args = p.parse_args()

    datasets = DIAG_DATASETS if args.dataset == "all" else [args.dataset]
    benchmark_configs = _benchmark_selected_rassg_configs(Path(args.config_csv))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for ds in datasets:
        print(f"=== {ds} ===")
        lam, m = _diagnostic_config(ds, args, benchmark_configs)
        print(f"  config: lam={lam:g} m={m}")
        split = make_synthetic(0) if ds == "synthetic" else load_dataset(ds)
        split = _limit_diagnostic_split(
            split,
            max_train=args.max_diagnostic_train,
            max_eval=args.max_diagnostic_eval,
            seed=0,
        )
        for seed in range(args.n_seeds):
            rassg_rows = run_rassg_diagnostics_dataset(
                split,
                ds,
                lam=lam,
                m=m,
                seed=seed,
            )
            max_rassg_grad_evals = max(
                (int(row["grad_evals"]) for row in rassg_rows), default=0
            )
            peg_rows = run_pegasos_diagnostics_dataset(
                split,
                ds,
                lam=lam,
                m=m,
                seed=seed,
                n_stages=args.n_stages,
                max_grad_evals=max_rassg_grad_evals,
            )
            all_rows.extend(rassg_rows)
            all_rows.extend(peg_rows)
            print(
                f"  seed={seed}  RASSG-r: {len(rassg_rows)} stages, Pegasos: {len(peg_rows)} stages"
            )

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DIAG_FIELDS)
        w.writeheader()
        w.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
