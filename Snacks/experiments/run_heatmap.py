#!/usr/bin/env python3
"""Heatmap experiment: RASSG-r test error vs (m, lambda) grid.

Fixes a dataset and runs RASSG-r over a grid of m (Nyström columns) and
lambda (regularization) values. Logs test error for each (m, lambda, seed)
to a CSV for heatmap visualization.

Usage:
    PYTHONPATH=src uv run python experiments/run_heatmap.py
    PYTHONPATH=src uv run python experiments/run_heatmap.py --dataset ijcnn1
    PYTHONPATH=src uv run python experiments/run_heatmap.py --smoke
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

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
M_GRID = [50, 75, 100, 150, 200, 300, 500, 750, 1_000, 1_500]
LAM_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
N_SEEDS = 3

FIELDS = [
    "dataset",
    "seed",
    "lam",
    "m",
    "n_train",
    "test_error",
    "test_acc",
]


def dense(arr) -> np.ndarray:
    return np.asarray(
        arr.toarray() if hasattr(arr, "toarray") else arr, dtype=np.float32
    )


def run_config(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    dataset: str,
    lam: float,
    m: int,
    seed: int,
) -> dict:
    """Run RASSG-r for one (m, lam, seed) configuration."""
    n_tr = len(y_tr)
    rng = np.random.default_rng(seed)

    gamma = float(
        median_bandwidth(
            X_tr, n_subsample=min(2000, n_tr), rng=np.random.default_rng(seed)
        )
    )
    columns = select_columns(X_tr, min(m, n_tr), rng)

    nys = NystromTransformer(
        lambda A, B: rbf_kernel(A, B, gamma), columns, ridge_mu=1e-6
    ).fit()
    Z_tr = nys.transform(X_tr)
    Z_val = nys.transform(X_val)
    Z_te = nys.transform(X_te)

    # Warm-up call.
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
        Z_test=Z_te,
        y_test=y_te,
        rng=np.random.default_rng(seed),
    )

    test_acc = float(np.mean((y_te * (Z_te @ res.u)) > 0))
    test_error = 1.0 - test_acc

    return {
        "dataset": dataset,
        "seed": seed,
        "lam": lam,
        "m": m,
        "n_train": n_tr,
        "test_error": test_error,
        "test_acc": test_acc,
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="tiny grid: m=[50,200], lam=[1e-3,1e-2], 1 seed",
    )
    p.add_argument("--out", default="experiments/results/heatmap_results.csv")
    args = p.parse_args()

    m_grid = [50, 200] if args.smoke else M_GRID
    lam_grid = [1e-3, 1e-2] if args.smoke else LAM_GRID
    n_seeds = 1 if args.smoke else N_SEEDS

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset}")
    split = load_dataset(args.dataset)
    X_tr = dense(split.X_train)
    X_val = dense(split.X_val)
    X_te = dense(split.X_test)
    print(f"n_train={split.n_train}  n_features={split.n_features}")
    print(f"m_grid={m_grid}  lam_grid={lam_grid}  seeds=0..{n_seeds - 1}")

    all_rows: list[dict] = []
    for m in m_grid:
        for lam in lam_grid:
            for seed in range(n_seeds):
                try:
                    row = run_config(
                        X_tr,
                        split.y_train,
                        X_te,
                        split.y_test,
                        X_val,
                        split.y_val,
                        dataset=args.dataset,
                        lam=lam,
                        m=m,
                        seed=seed,
                    )
                    all_rows.append(row)
                    print(
                        f"  m={m:>5} lam={lam:.0e} seed={seed}: test_error={row['test_error']:.4f}"
                    )
                except Exception as e:
                    print(f"  m={m} lam={lam} seed={seed}: FAILED: {e}")

    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
