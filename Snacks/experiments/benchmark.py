#!/usr/bin/env python3
"""Benchmark RASSG-r vs LibLinear vs Pegasos vs sklearn-Nystrom.

RASSG-r, LibLinear, and Pegasos share the same Nystrom features. sklearn-Nystrom
runs its own embedding (end-to-end baseline). Solver-only and end-to-end
timings are reported separately.

Usage:
    PYTHONPATH=src uv run python experiments/benchmark.py --dataset synthetic --smoke
    PYTHONPATH=src uv run python experiments/benchmark.py --dataset mushrooms
    PYTHONPATH=src uv run python experiments/benchmark.py --dataset all
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import re
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from lib.common import (
    RASSG_R_DEFAULT_BETA0_SCALE,
    RASSG_R_DEFAULT_BETA_DECAY,
    RASSG_R_DEFAULT_GROWTH,
    RASSG_R_DEFAULT_M_INNER0,
    RASSG_R_DEFAULT_RESTARTS,
    RASSG_R_DEFAULT_STAGES_PER_RESTART,
    Timer,
    append_rows,
    default_assg_m_inner,
    default_nystrom_m_grid,
    empty_row,
    make_run_id,
    parse_int_grid,
    save_metadata,
)
from lib.datasets import Split, load_dataset, _REGISTRY

from snacks.kernels import median_bandwidth, rbf_kernel
from snacks.nystrom import NystromTransformer, select_columns
from snacks.solver import rassg_r

SCRIPT_NAME = "benchmark.py"
RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_RAW_RESULTS = RESULTS_DIR / "raw_results.csv"
DEFAULT_ARCHIVED_RAW_RESULTS = RESULTS_DIR / "old_raw_results.csv"
SOLVERS = ("RASSG-r", "LibLinear", "Pegasos", "sklearn-Nystrom")
SHARED_FEATURE_SOLVERS = frozenset({"RASSG-r", "LibLinear", "Pegasos"})
SOLVER_ALIASES = {"Snacks": "RASSG-r"}
BENCHMARK_DATASETS = ("mushrooms", "a1a", "madelon", "news20binary")
ROUND_M_CANDIDATES = (
    50,
    100,
    200,
    500,
    1_000,
    2_000,
    5_000,
    10_000,
    20_000,
    50_000,
    100_000,
    120_000,
)
DEFAULT_POOR_ACCURACY_THRESHOLD = 0.8
DEFAULT_POOR_ACCURACY_MIN_TRAIN = 100_000
DEFAULT_PEAK_SAFETY_FACTOR = 1.1
TRANSFORM_CHUNK_TMP_FACTOR = 6
_MALLOC_TRIM = None

# Validation-selected RASSG-r lambdas from experiments/results/old_raw_results.csv.
# Selection is by mean validation accuracy over seeds at the best available m.
VALIDATION_SELECTED_LAMBDAS: dict[str, float] = {
    "mushrooms": 1e-4,
    "a1a": 1e-4,
    "splice": 1e-4,
    "w8a": 1e-4,
    "ijcnn1": 1e-4,
    "madelon": 1e-4,
    "mnist": 1e-4,
    "covtype.binary": 1e-4,
    "epsilon": 1e-4,
    "SUSY": 1e-4,
    "HIGGS": 1e-4,
    "YearPredictionMSD": 1e-4,
    "news20binary": 1e-4,
}
DEFAULT_SYNTHETIC_LAM = 1e-3

# Per-dataset m-grid overrides for recovery runs. Lambdas are selected
# separately through VALIDATION_SELECTED_LAMBDAS, not swept here.
RECOVERY_GRIDS: dict[str, dict] = {
    "splice": {"m_grid": [274]},
    "w8a": {"m_grid": [2711]},
    "ijcnn1": {"m_grid": [1360, 2719]},
}


def _solver_row_params(solver: str, n_stages: int, m_inner: int) -> tuple[int, int]:
    if solver == "RASSG-r":
        return (
            RASSG_R_DEFAULT_RESTARTS * RASSG_R_DEFAULT_STAGES_PER_RESTART,
            RASSG_R_DEFAULT_M_INNER0,
        )
    return n_stages, m_inner


def _float_key(value: object) -> str:
    return f"{float(value):.17g}"


def _entry_key(
    dataset: str,
    solver: str,
    split_seed: int,
    solver_seed: int,
    lam: float,
    m: int,
    n_stages: int,
    m_inner: int,
    feature_dtype: str = "float32",
) -> tuple[str, str, str, int, int, str, int, int, int, str]:
    return (
        SCRIPT_NAME,
        dataset,
        solver,
        int(split_seed),
        int(solver_seed),
        _float_key(lam),
        int(m),
        int(n_stages),
        int(m_inner),
        feature_dtype,
    )


def _row_feature_dtype(row: dict) -> str:
    notes = row.get("notes", "")
    match = re.search(r"(?:^|;\s*)feature_dtype=([^;]+)", notes)
    if match:
        return match.group(1)
    # Rows produced before feature-dtype support are not considered complete for
    # the current float32-only benchmark.
    return "legacy"


def _row_entry_key(
    row: dict,
) -> tuple[str, str, str, int, int, str, int, int, int, str] | None:
    try:
        if row.get("script_name") != SCRIPT_NAME:
            return None
        return _entry_key(
            str(row["dataset"]),
            str(row["solver"]),
            int(row["split_seed"]),
            int(row["solver_seed"]),
            float(row["lam"]),
            int(row["m"]),
            int(row["n_stages"]),
            int(row["m_inner"]),
            _row_feature_dtype(row),
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_completed_entries(
    out_path: Path,
) -> set[tuple[str, str, str, int, int, str, int, int, int, str]]:
    if not out_path.exists():
        return set()
    with open(out_path, newline="") as f:
        return {
            key for row in csv.DictReader(f) if (key := _row_entry_key(row)) is not None
        }


def missing_solvers_for_config(
    completed_entries: set[tuple[str, str, str, int, int, str, int, int, int, str]],
    dataset: str,
    lam: float,
    m: int,
    seed: int,
    n_stages: int,
    m_inner: int,
    feature_dtype: str,
    solvers: Iterable[str],
) -> list[str]:
    missing = []
    for solver in solvers:
        solver_n_stages, solver_m_inner = _solver_row_params(solver, n_stages, m_inner)
        key = _entry_key(
            dataset,
            solver,
            seed,
            seed,
            lam,
            m,
            solver_n_stages,
            solver_m_inner,
            feature_dtype,
        )
        if key not in completed_entries:
            missing.append(solver)
    return missing


def _normalize_solvers(solvers: Iterable[str] | None) -> list[str]:
    if solvers is None:
        return list(SOLVERS)
    selected = {SOLVER_ALIASES.get(solver, solver) for solver in solvers}
    unknown = selected.difference(SOLVERS)
    if unknown:
        raise ValueError(f"Unknown solver(s): {', '.join(sorted(unknown))}")
    return [solver for solver in SOLVERS if solver in selected]


def _parse_float_grid(value: str) -> list[float]:
    grid = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        item = float(part)
        if item <= 0:
            raise ValueError("grid values must be positive.")
        grid.append(item)
    if not grid:
        raise ValueError("grid must contain at least one value.")
    return sorted(set(grid))


def _f(s) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _feature_dtype(args: argparse.Namespace):
    return np.float32


def _release_unused_memory() -> None:
    """Return large freed NumPy/sklearn allocations to the OS when possible."""
    gc.collect()
    global _MALLOC_TRIM
    if not sys.platform.startswith("linux"):
        return
    if _MALLOC_TRIM is None:
        try:
            _MALLOC_TRIM = ctypes.CDLL(None).malloc_trim
            _MALLOC_TRIM.argtypes = [ctypes.c_size_t]
            _MALLOC_TRIM.restype = ctypes.c_int
        except (AttributeError, OSError):
            _MALLOC_TRIM = False
    if _MALLOC_TRIM:
        _MALLOC_TRIM(0)


def _auto_lam_grid_for_dataset(dataset: str) -> list[float]:
    if dataset == "synthetic":
        return [DEFAULT_SYNTHETIC_LAM]
    if dataset not in VALIDATION_SELECTED_LAMBDAS:
        raise ValueError(
            f"No validation-selected lambda configured for {dataset!r}. "
            "Add it to VALIDATION_SELECTED_LAMBDAS or pass --lam-grid explicitly."
        )
    return [VALIDATION_SELECTED_LAMBDAS[dataset]]


def _validation_selected_primary(path: Path) -> dict[str, dict]:
    """Validation-selected RASSG-r stats from an existing raw benchmark CSV."""
    if not path.exists():
        return {}

    by_cfg: dict[tuple[str, str, str], list[dict]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("script_name") != SCRIPT_NAME or row.get("solver") != "RASSG-r":
                continue
            val = _f(row.get("val_acc"))
            if val is None:
                continue
            key = (row["dataset"], row.get("lam", ""), row.get("m", ""))
            by_cfg.setdefault(key, []).append(row)

    out: dict[str, dict] = {}
    for (dataset, lam_s, m_s), rows in by_cfg.items():
        vals = [_f(r.get("val_acc")) for r in rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        tests = [_f(r.get("test_acc")) for r in rows]
        tests = [v for v in tests if v is not None]
        n_train = _f(rows[0].get("n_train"))
        entry = {
            "lam": _f(lam_s),
            "m": _f(m_s),
            "val_acc": float(np.mean(vals)),
            "test_acc": float(np.mean(tests)) if tests else None,
            "n_train": int(n_train) if n_train is not None else 0,
        }
        if entry["lam"] is None or entry["m"] is None:
            continue
        if dataset not in out or entry["val_acc"] > out[dataset]["val_acc"]:
            out[dataset] = entry
    return out


def _validation_selected_reference(arg_value: str) -> dict[str, dict]:
    path = Path(arg_value)
    selected = _validation_selected_primary(path)
    if arg_value == str(DEFAULT_RAW_RESULTS) and DEFAULT_ARCHIVED_RAW_RESULTS.exists():
        archived = _validation_selected_primary(DEFAULT_ARCHIVED_RAW_RESULTS)
        if not path.exists():
            if archived:
                print(
                    f"Using archived poor-accuracy reference CSV: {DEFAULT_ARCHIVED_RAW_RESULTS}"
                )
            return archived
        n_added = 0
        for dataset, entry in archived.items():
            if dataset not in selected:
                selected[dataset] = entry
                n_added += 1
        if n_added:
            print(
                f"Supplemented poor-accuracy references for {n_added} missing datasets "
                f"from {DEFAULT_ARCHIVED_RAW_RESULTS}"
            )
    return selected


def _poor_accuracy_dataset(
    dataset: str, selected: dict[str, dict], args: argparse.Namespace
) -> bool:
    entry = selected.get(dataset)
    if entry is None:
        return False
    if entry.get("n_train", 0) < args.poor_accuracy_min_train:
        return False
    return entry["val_acc"] < args.poor_accuracy_threshold


def _round_m_grid_for_split(
    split: Split,
    args: argparse.Namespace,
    solvers: Iterable[str],
) -> list[int]:
    budget = args.max_peak_gb if args.max_peak_gb > 0 else 24.0
    feature_dtype = _feature_dtype(args)
    grid = [
        m
        for m in ROUND_M_CANDIDATES
        if m <= split.n_train
        and estimate_peak_gib(
            split,
            m,
            args.transform_chunk_mb,
            feature_dtype=feature_dtype,
            solvers=solvers,
            peak_safety_factor=args.peak_safety_factor,
        )
        <= budget
    ]
    if not grid:
        raise ValueError(
            f"No round m candidate fits the {budget:.3g} GiB peak budget. "
            "Increase --max-peak-gb or provide --m-grid explicitly."
        )
    return grid


def _poor_accuracy_m_grid(
    split: Split,
    args: argparse.Namespace,
    solvers: Iterable[str],
) -> list[int]:
    selected = _normalize_solvers(solvers)
    return [max(_round_m_grid_for_split(split, args, selected))]


def _ceil_round_m(value: int, candidates: list[int]) -> int:
    for m in candidates:
        if m >= value:
            return m
    return candidates[-1]


def _rounded_default_m_grid(
    split: Split,
    args: argparse.Namespace,
    solvers: Iterable[str],
) -> list[int]:
    feasible = _round_m_grid_for_split(split, args, solvers)
    default_grid = default_nystrom_m_grid(
        split.n_train,
        delta=args.nystrom_delta,
        eigen_decay_p=args.nystrom_eigen_decay_p,
        max_feature_mb=args.max_feature_mb,
    )
    return sorted({_ceil_round_m(m, feasible) for m in default_grid})


def _m_grid_for_split(
    split: Split,
    dataset: str,
    smoke: bool,
    args: argparse.Namespace,
    solvers: Iterable[str],
    selected_primary: dict[str, dict],
) -> list[int]:
    if args.m_grid != "auto":
        if args.m_grid == "auto-round":
            grid = _rounded_default_m_grid(split, args, solvers)
            if _poor_accuracy_dataset(dataset, selected_primary, args):
                return _poor_accuracy_m_grid(split, args, solvers)
            return grid
        if args.m_grid == "recovery":
            if dataset not in RECOVERY_GRIDS:
                return _rounded_default_m_grid(split, args, solvers)
            return RECOVERY_GRIDS[dataset]["m_grid"]
        return parse_int_grid(args.m_grid)
    if smoke:
        return [50]
    grid = _rounded_default_m_grid(split, args, solvers)
    if _poor_accuracy_dataset(dataset, selected_primary, args):
        return _poor_accuracy_m_grid(split, args, solvers)
    return grid


def _known_m_grid_before_load(
    dataset: str,
    smoke: bool,
    args: argparse.Namespace,
    selected_primary: dict[str, dict],
) -> list[int] | None:
    if smoke:
        return [50]
    if args.m_grid == "auto":
        if _poor_accuracy_dataset(dataset, selected_primary, args):
            return [int(selected_primary[dataset]["m"])]
        return None
    if args.m_grid == "auto-round":
        if _poor_accuracy_dataset(dataset, selected_primary, args):
            return [int(selected_primary[dataset]["m"])]
        return None
    if args.m_grid == "recovery":
        if dataset in RECOVERY_GRIDS:
            return RECOVERY_GRIDS[dataset]["m_grid"]
        return None
    return parse_int_grid(args.m_grid)


def _all_rows_complete_for_known_grid(
    completed_entries: set[tuple[str, str, str, int, int, str, int, int, int, str]],
    dataset: str,
    lam_grid: Iterable[float],
    m_grid: Iterable[int],
    seeds: Iterable[int],
    n_stages: int,
    m_inner: int,
    feature_dtype: str,
    solvers: Iterable[str],
) -> bool:
    for lam in lam_grid:
        for m in m_grid:
            for seed in seeds:
                missing = missing_solvers_for_config(
                    completed_entries,
                    dataset,
                    lam,
                    m,
                    seed,
                    n_stages,
                    m_inner,
                    feature_dtype,
                    solvers,
                )
                if missing:
                    return False
    return True


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


def dense(arr, dtype=None) -> np.ndarray:
    return np.asarray(arr.toarray() if hasattr(arr, "toarray") else arr, dtype=dtype)


def densify_split(split: Split, dtype=np.float32) -> Split:
    """Materialize dense arrays once, releasing sparse split matrices as we go."""
    split.X_train = dense(split.X_train, dtype=dtype)
    gc.collect()
    split.X_val = dense(split.X_val, dtype=dtype)
    gc.collect()
    split.X_test = dense(split.X_test, dtype=dtype)
    gc.collect()
    return split


def _chunk_rows(n_components: int, chunk_mb: float, itemsize: int = 8) -> int | None:
    if chunk_mb <= 0:
        return None
    # RBF evaluation and the following matrix multiply materialize several
    # chunk-sized arrays. Keep the row chunk smaller than the nominal cap.
    bytes_per_row = max(1, TRANSFORM_CHUNK_TMP_FACTOR * n_components * itemsize)
    return max(1, int(chunk_mb * 1024 * 1024 // bytes_per_row))


def transform_chunked(
    transformer, X: np.ndarray, chunk_mb: float, output_dtype=None
) -> np.ndarray:
    """Transform X in row chunks to cap temporary kernel memory."""
    n = X.shape[0]
    n_components = (
        transformer.columns.shape[0]
        if hasattr(transformer, "columns")
        else transformer.n_components
    )
    desired_dtype = (
        output_dtype
        if output_dtype is not None
        else getattr(transformer, "output_dtype", None)
    )
    desired_dtype = np.dtype(desired_dtype) if desired_dtype is not None else None
    itemsize = desired_dtype.itemsize if desired_dtype is not None else X.dtype.itemsize
    rows_per_chunk = _chunk_rows(n_components, chunk_mb, itemsize)
    if rows_per_chunk is None or rows_per_chunk >= n:
        out = transformer.transform(X)
        return (
            out.astype(desired_dtype, copy=False) if desired_dtype is not None else out
        )

    first_end = min(rows_per_chunk, n)
    first = transformer.transform(X[:first_end])
    if desired_dtype is not None:
        first = first.astype(desired_dtype, copy=False)
    Z = np.empty((n, first.shape[1]), dtype=first.dtype)
    Z[:first_end] = first
    del first
    for start in range(first_end, n, rows_per_chunk):
        end = min(start + rows_per_chunk, n)
        chunk = transformer.transform(X[start:end])
        if desired_dtype is not None:
            chunk = chunk.astype(desired_dtype, copy=False)
        Z[start:end] = chunk
        del chunk
    return Z


def estimate_peak_gib(
    split: Split,
    m: int,
    transform_chunk_mb: float,
    *,
    feature_dtype=np.float32,
    solvers: Iterable[str] | None = None,
    peak_safety_factor: float = 1.0,
) -> float:
    """Conservative peak estimate for one benchmark config after densification."""
    if peak_safety_factor <= 0:
        raise ValueError("peak_safety_factor must be positive.")
    selected_solvers = _normalize_solvers(solvers)
    m_eff = min(m, split.n_train)
    n_train = split.X_train.shape[0]
    n_val = split.X_val.shape[0]
    n_test = split.X_test.shape[0]
    feature_itemsize = np.dtype(feature_dtype).itemsize
    kernel_itemsize = max(split.X_train.dtype.itemsize, feature_itemsize)
    dense_x = split.X_train.nbytes + split.X_val.nbytes + split.X_test.nbytes
    z_train = n_train * m_eff * feature_itemsize
    z_val = n_val * m_eff * feature_itemsize
    z_test = n_test * m_eff * feature_itemsize
    shared_z = z_train + z_val + z_test
    sklearn_train_copy = n_train * m_eff * np.dtype(np.float64).itemsize
    eig_workspace = 3 * m_eff * m_eff * feature_itemsize
    rows_per_chunk = _chunk_rows(m_eff, transform_chunk_mb, kernel_itemsize)
    if rows_per_chunk is None:
        chunk_tmp = TRANSFORM_CHUNK_TMP_FACTOR * max(z_train, z_val, z_test)
    else:
        chunk_tmp = (
            min(max(n_train, n_val, n_test), rows_per_chunk)
            * m_eff
            * kernel_itemsize
            * TRANSFORM_CHUNK_TMP_FACTOR
        )

    phases = [dense_x]
    if set(selected_solvers).intersection(SHARED_FEATURE_SOLVERS):
        shared_solver_copy = (
            sklearn_train_copy if "LibLinear" in selected_solvers else 0
        )
        phases.append(
            dense_x + shared_z + eig_workspace + chunk_tmp + shared_solver_copy
        )
    if "sklearn-Nystrom" in selected_solvers:
        phases.append(
            dense_x + z_train + sklearn_train_copy + eig_workspace + chunk_tmp
        )
    return peak_safety_factor * max(phases) / (1024**3)


def _memory_feasible_solvers(
    split: Split,
    m: int,
    args: argparse.Namespace,
    feature_dtype,
    solvers: Iterable[str],
) -> tuple[list[str], list[tuple[str, float]]]:
    selected = _normalize_solvers(solvers)
    if args.max_peak_gb <= 0:
        return selected, []

    feasible: list[str] = []
    skipped: list[tuple[str, float]] = []
    for solver in selected:
        peak_gib = estimate_peak_gib(
            split,
            m,
            args.transform_chunk_mb,
            feature_dtype=feature_dtype,
            solvers=[solver],
            peak_safety_factor=args.peak_safety_factor,
        )
        if peak_gib > args.max_peak_gb:
            skipped.append((solver, peak_gib))
        else:
            feasible.append(solver)
    return feasible, skipped


def run_config(
    split: Split,
    lam: float,
    m: int,
    seed: int,
    n_stages: int,
    m_inner: int,
    run_id: str = "",
    dataset: str = "",
    transform_chunk_mb: float = 512.0,
    feature_dtype=np.float32,
    solvers: Iterable[str] | None = None,
) -> list[dict]:
    """Run selected solvers on one (lam, m, seed) configuration."""
    selected_solvers = _normalize_solvers(solvers)
    if not selected_solvers:
        return []
    shared_solvers = set(selected_solvers).intersection(SHARED_FEATURE_SOLVERS)

    feature_dtype = np.dtype(feature_dtype)
    rng = np.random.default_rng(seed)
    X_tr, X_val, X_te = dense(split.X_train), dense(split.X_val), dense(split.X_test)
    y_tr, y_val, y_te = split.y_train, split.y_val, split.y_test
    n_tr = len(y_tr)

    gamma = float(
        median_bandwidth(X_tr, n_subsample=2000, rng=np.random.default_rng(seed))
    )
    kernel = lambda A, B: rbf_kernel(A, B, gamma)  # noqa: E731

    base = dict(
        run_id=run_id,
        script_name=SCRIPT_NAME,
        dataset=dataset,
        split_seed=seed,
        solver_seed=seed,
        lam=lam,
        m=m,
        gamma=gamma,
        ridge_mu=1e-6,
        n_train=n_tr,
        n_val=len(y_val),
        n_test=len(y_te),
        n_features=split.n_features,
        is_sparse=split.is_sparse,
        n_stages=n_stages,
        m_inner=m_inner,
    )

    rows: list[dict] = []

    if shared_solvers:
        # Fitted Nyström transformer: one eigendecomposition shared across splits.
        columns = select_columns(X_tr, min(m, n_tr), rng)
        with Timer() as t_fit:
            nys = NystromTransformer(
                kernel,
                columns,
                ridge_mu=1e-6,
                output_dtype=feature_dtype,
            ).fit()
        nystrom_fit_time = t_fit.elapsed
        with Timer() as t_tr_z:
            Z_tr = transform_chunked(nys, X_tr, transform_chunk_mb)
        with Timer() as t_val_z:
            Z_val = transform_chunked(nys, X_val, transform_chunk_mb)
        with Timer() as t_te_z:
            Z_te = transform_chunked(nys, X_te, transform_chunk_mb)

    if "RASSG-r" in selected_solvers:
        # --- RASSG-r on shared Z (solver-only) ---
        # Warmup outside of timing.
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
                y_val=y_val,
                rng=np.random.default_rng(seed),
            )
        with Timer() as t_pred:
            preds = (Z_te @ res.u) > 0
        optimizer_time = (
            res.optimizer_time if res.optimizer_time is not None else t_solver.elapsed
        )
        selection_time = (
            res.selection_time
            if res.selection_time is not None
            else max(0.0, t_solver.elapsed - optimizer_time)
        )
        train_acc = float(np.mean(((Z_tr @ res.u) > 0) == (y_tr > 0)))
        test_acc = float(np.mean(preds == (y_te > 0)))
        val_acc = float(np.mean(((Z_val @ res.u) > 0) == (y_val > 0)))
        rassg_base = {
            **base,
            "n_stages": RASSG_R_DEFAULT_RESTARTS * RASSG_R_DEFAULT_STAGES_PER_RESTART,
            "m_inner": RASSG_R_DEFAULT_M_INNER0,
        }
        rows.append(
            empty_row(
                **rassg_base,
                solver="RASSG-r",
                alpha=0.5,
                q=16,
                adaptive_decay=False,
                val_acc=val_acc,
                train_acc=train_acc,
                test_acc=test_acc,
                train_time_solver=optimizer_time,
                predict_time_solver=t_pred.elapsed,
                nystrom_fit_time=nystrom_fit_time,
                nystrom_transform_train_time=t_tr_z.elapsed,
                nystrom_transform_val_time=t_val_z.elapsed,
                nystrom_transform_test_time=t_te_z.elapsed,
                total_train_time=nystrom_fit_time + t_tr_z.elapsed + t_solver.elapsed,
                total_predict_time=t_te_z.elapsed + t_pred.elapsed,
                total_time=(
                    nystrom_fit_time
                    + t_tr_z.elapsed
                    + t_te_z.elapsed
                    + t_solver.elapsed
                    + t_pred.elapsed
                ),
                n_stages_run=res.n_stages_run,
                stop_reason=res.stop_reason,
                notes=(
                    "solver-only timing on shared Nystrom features; warmup excluded; "
                    f"restarts={RASSG_R_DEFAULT_RESTARTS}; "
                    f"stages_per_restart={RASSG_R_DEFAULT_STAGES_PER_RESTART}; "
                    f"m_inner0={RASSG_R_DEFAULT_M_INNER0}; growth={RASSG_R_DEFAULT_GROWTH:g}; "
                    f"beta0_scale={RASSG_R_DEFAULT_BETA0_SCALE:g}; "
                    f"beta_decay={RASSG_R_DEFAULT_BETA_DECAY:g}; "
                    f"validation_selection_time={selection_time:.6g}; "
                    f"optimizer_wall_time={optimizer_time:.6g}; "
                    "selected by validation accuracy; "
                    f"raw_dtype={X_tr.dtype.name}; feature_dtype={feature_dtype.name}"
                ),
            )
        )

    if "LibLinear" in selected_solvers:
        # --- LibLinear on shared Z ---
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
            preds = clf.predict(Z_te)
        train_acc = float(np.mean(clf.predict(Z_tr) == y_tr))
        test_acc = float(np.mean(preds == y_te))
        val_acc = float(np.mean(clf.predict(Z_val) == y_val))
        rows.append(
            empty_row(
                **base,
                solver="LibLinear",
                val_acc=val_acc,
                train_acc=train_acc,
                test_acc=test_acc,
                train_time_solver=t_solver.elapsed,
                predict_time_solver=t_pred.elapsed,
                nystrom_fit_time=nystrom_fit_time,
                nystrom_transform_train_time=t_tr_z.elapsed,
                nystrom_transform_val_time=t_val_z.elapsed,
                nystrom_transform_test_time=t_te_z.elapsed,
                total_train_time=nystrom_fit_time + t_tr_z.elapsed + t_solver.elapsed,
                total_predict_time=t_te_z.elapsed + t_pred.elapsed,
                total_time=(
                    nystrom_fit_time
                    + t_tr_z.elapsed
                    + t_te_z.elapsed
                    + t_solver.elapsed
                    + t_pred.elapsed
                ),
                notes=(
                    "solver on shared Nystrom features; "
                    f"raw_dtype={X_tr.dtype.name}; feature_dtype={feature_dtype.name}; "
                    f"input_feature_dtype={Z_tr.dtype.name}; coef_dtype={clf.coef_.dtype.name}"
                ),
            )
        )

    if "Pegasos" in selected_solvers:
        # --- Pegasos (SGDClassifier) on shared Z ---
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
        with Timer() as t_solver:
            clf.fit(Z_tr, y_tr)
        with Timer() as t_pred:
            preds = clf.predict(Z_te)
        train_acc = float(np.mean(clf.predict(Z_tr) == y_tr))
        test_acc = float(np.mean(preds == y_te))
        val_acc = float(np.mean(clf.predict(Z_val) == y_val))
        rows.append(
            empty_row(
                **base,
                solver="Pegasos",
                val_acc=val_acc,
                train_acc=train_acc,
                test_acc=test_acc,
                train_time_solver=t_solver.elapsed,
                predict_time_solver=t_pred.elapsed,
                nystrom_fit_time=nystrom_fit_time,
                nystrom_transform_train_time=t_tr_z.elapsed,
                nystrom_transform_val_time=t_val_z.elapsed,
                nystrom_transform_test_time=t_te_z.elapsed,
                total_train_time=nystrom_fit_time + t_tr_z.elapsed + t_solver.elapsed,
                total_predict_time=t_te_z.elapsed + t_pred.elapsed,
                total_time=(
                    nystrom_fit_time
                    + t_tr_z.elapsed
                    + t_te_z.elapsed
                    + t_solver.elapsed
                    + t_pred.elapsed
                ),
                notes=(
                    "solver on shared Nystrom features; "
                    f"raw_dtype={X_tr.dtype.name}; feature_dtype={feature_dtype.name}; "
                    f"input_feature_dtype={Z_tr.dtype.name}; coef_dtype={clf.coef_.dtype.name}"
                ),
            )
        )

    if shared_solvers:
        # The sklearn-Nystrom baseline builds its own dense embedding. Release the
        # shared embedding first so we do not keep two full n x m feature matrices
        # alive at once.
        del Z_tr, Z_val, Z_te, nys
        _release_unused_memory()

    if "sklearn-Nystrom" in selected_solvers:
        # --- sklearn-Nystrom (own embedding, end-to-end) ---
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
        with Timer() as t_train_e2e:
            sk_nys.fit(X_tr)
            Z_tr_sk = transform_chunked(
                sk_nys, X_tr, transform_chunk_mb, output_dtype=feature_dtype
            )
            clf.fit(Z_tr_sk, y_tr)
            train_acc = float(np.mean(clf.predict(Z_tr_sk) == y_tr))
        del Z_tr_sk
        with Timer() as t_pred_e2e:
            Z_te_sk = transform_chunked(
                sk_nys, X_te, transform_chunk_mb, output_dtype=feature_dtype
            )
            preds = clf.predict(Z_te_sk)
            test_acc = float(np.mean(preds == y_te))
            del Z_te_sk
            Z_val_sk = transform_chunked(
                sk_nys, X_val, transform_chunk_mb, output_dtype=feature_dtype
            )
            val_acc = float(np.mean(clf.predict(Z_val_sk) == y_val))
            del Z_val_sk
        rows.append(
            empty_row(
                **base,
                solver="sklearn-Nystrom",
                val_acc=val_acc,
                train_acc=train_acc,
                test_acc=test_acc,
                total_train_time=t_train_e2e.elapsed,
                total_predict_time=t_pred_e2e.elapsed,
                total_time=t_train_e2e.elapsed + t_pred_e2e.elapsed,
                notes=(
                    "end-to-end timing including own Nystrom embedding; "
                    f"raw_dtype={X_tr.dtype.name}; feature_dtype={feature_dtype.name}; "
                    f"input_feature_dtype={feature_dtype.name}; coef_dtype={clf.coef_.dtype.name}"
                ),
            )
        )
        del sk_nys, clf
        _release_unused_memory()

    # Compatibility aliases used by tests/test_smoke.py.
    for r in rows:
        r["accuracy"] = r["test_acc"]
        if r["solver"] == "sklearn-Nystrom":
            r["train_time"] = r["total_train_time"]
            r["predict_time"] = r["total_predict_time"]
        else:
            r["train_time"] = r["train_time_solver"]
            r["predict_time"] = r["predict_time_solver"]
        r["seed"] = r["solver_seed"]

    return rows


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dataset",
        default="all",
        help=f"'synthetic', 'all', or one of {BENCHMARK_DATASETS}",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny grid (lam=1e-3, m=50, 1 seed, 5 stages)",
    )
    p.add_argument("--out", default="experiments/results/raw_results.csv")
    p.add_argument(
        "--m-grid",
        default="auto",
        help="Comma-separated positive m values, 'auto'/'auto-round' for rounded automatic values "
        "that fit --max-peak-gb, or 'recovery' for the old crash-recovery grid",
    )
    p.add_argument(
        "--lam-grid",
        default="auto",
        help="Comma-separated positive lambda values, or 'auto' for the fixed per-dataset defaults",
    )
    p.add_argument(
        "--max-feature-mb",
        type=float,
        default=1024.0,
        help="Auto m-grid cap for dense train Nyström features Z_train",
    )
    p.add_argument(
        "--transform-chunk-mb",
        type=float,
        default=512.0,
        help="Approximate temporary-memory cap for each Nyström transform chunk. "
        "Use <=0 to disable chunking.",
    )
    p.add_argument(
        "--max-peak-gb",
        type=float,
        default=24.0,
        help="Skip configurations whose estimated peak RAM exceeds this many GiB. "
        "Use 0 to disable skipping.",
    )
    p.add_argument(
        "--peak-safety-factor",
        type=float,
        default=DEFAULT_PEAK_SAFETY_FACTOR,
        help="Multiplier applied to estimated peak memory before comparing with "
        "--max-peak-gb.",
    )
    p.add_argument(
        "--raw-dtype",
        choices=["float32"],
        default="float32",
        help="dtype used to hold dense raw X in this benchmark.",
    )
    p.add_argument(
        "--feature-dtype",
        choices=["float32"],
        default="float32",
        help="dtype used to store shared Nyström features.",
    )
    p.add_argument(
        "--solvers",
        default=",".join(SOLVERS),
        help=f"Comma-separated subset of {SOLVERS}.",
    )
    p.add_argument(
        "--poor-accuracy-threshold",
        type=float,
        default=DEFAULT_POOR_ACCURACY_THRESHOLD,
        help="Large datasets with validation-selected RASSG-r val_acc below this use the "
        "largest selected-solver-feasible round m when --m-grid is auto/auto-round.",
    )
    p.add_argument(
        "--poor-accuracy-min-train",
        type=int,
        default=DEFAULT_POOR_ACCURACY_MIN_TRAIN,
        help="Minimum n_train for the poor-accuracy largest-m policy.",
    )
    p.add_argument(
        "--poor-accuracy-csv",
        default=str(DEFAULT_RAW_RESULTS),
        help="Raw results CSV used to identify poor-accuracy datasets. "
        "If the default file is absent, old_raw_results.csv is used when present.",
    )
    p.add_argument(
        "--nystrom-delta",
        type=float,
        default=0.05,
        help="Failure-probability scale used by the auto m-grid",
    )
    p.add_argument(
        "--nystrom-eigen-decay-p",
        type=float,
        default=0.5,
        help="Polynomial eigendecay exponent p used by the auto m-grid",
    )
    p.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Run all scheduled solver rows even when matching rows already exist in --out.",
    )
    args = p.parse_args()
    if args.peak_safety_factor <= 0:
        raise ValueError("--peak-safety-factor must be positive.")

    if args.smoke:
        lam_grid, seeds, n_stages, m_inner = [1e-3], [0], 5, 128
    else:
        lam_grid = None if args.lam_grid == "auto" else _parse_float_grid(args.lam_grid)
        seeds, n_stages, m_inner = [0, 1, 2], 20, None

    datasets = list(BENCHMARK_DATASETS) if args.dataset == "all" else [args.dataset]
    out_path = Path(args.out)
    run_id = make_run_id()
    raw_dtype = np.float32
    feature_dtype = _feature_dtype(args)
    selected_solvers = _normalize_solvers(
        s.strip() for s in args.solvers.split(",") if s.strip()
    )
    selected_primary = _validation_selected_reference(args.poor_accuracy_csv)
    completed_entries = (
        set() if args.rerun_existing else load_completed_entries(out_path)
    )
    if completed_entries:
        print(
            f"Found {len(completed_entries)} existing benchmark rows in {out_path}; running missing rows only."
        )

    total_rows = 0
    total_skipped = 0
    total_memory_skipped = 0
    for ds in datasets:
        _release_unused_memory()
        print(f"\n=== {ds} ===")
        if args.smoke:
            ds_lam_grid = lam_grid
        elif lam_grid is None:
            ds_lam_grid = _auto_lam_grid_for_dataset(ds)
        else:
            ds_lam_grid = lam_grid
        known_m_grid = _known_m_grid_before_load(ds, args.smoke, args, selected_primary)
        known_n_train = int(selected_primary.get(ds, {}).get("n_train", 1))
        known_inner = m_inner if m_inner else default_assg_m_inner(known_n_train)
        if (
            not args.rerun_existing
            and known_m_grid is not None
            and _all_rows_complete_for_known_grid(
                completed_entries,
                ds,
                ds_lam_grid,
                known_m_grid,
                seeds,
                n_stages,
                known_inner,
                np.dtype(feature_dtype).name,
                selected_solvers,
            )
        ):
            skipped = (
                len(ds_lam_grid)
                * len(known_m_grid)
                * len(seeds)
                * len(selected_solvers)
            )
            total_skipped += skipped
            print(
                f"SKIP dataset before load: lambda_values={ds_lam_grid}, "
                f"m_grid={known_m_grid} already complete"
            )
            continue

        split = (
            make_synthetic() if ds == "synthetic" else load_dataset(ds, dtype=raw_dtype)
        )
        split = densify_split(split, dtype=raw_dtype)
        _release_unused_memory()
        print(f"n_train={split.n_train} n_features={split.n_features}")
        use_poor_accuracy_max_m = _poor_accuracy_dataset(ds, selected_primary, args)
        m_grid = _m_grid_for_split(
            split, ds, args.smoke, args, selected_solvers, selected_primary
        )
        if use_poor_accuracy_max_m:
            source = selected_primary[ds]
            lam_policy = (
                "fixed dataset lambda"
                if args.lam_grid == "auto"
                else "requested lambda values"
            )
            print(
                f"poor-accuracy policy: previous RASSG-r val_acc={source['val_acc']:.4f} "
                f"at lam={source['lam']:.0e}, m={int(source['m'])}; "
                f"using {lam_policy} and largest feasible round m for selected solvers"
            )
        print(f"lambda_values={ds_lam_grid}, m_grid={m_grid}")
        for lam in ds_lam_grid:
            for m in m_grid:
                for seed in seeds:
                    inner = m_inner if m_inner else default_assg_m_inner(split.n_train)
                    solvers_to_run = (
                        list(selected_solvers)
                        if args.rerun_existing
                        else missing_solvers_for_config(
                            completed_entries,
                            ds,
                            lam,
                            m,
                            seed,
                            n_stages,
                            inner,
                            feature_dtype=np.dtype(feature_dtype).name,
                            solvers=selected_solvers,
                        )
                    )
                    if not solvers_to_run:
                        total_skipped += len(selected_solvers)
                        print(
                            f"  SKIP lam={lam:.0e} m={m} seed={seed}: already complete"
                        )
                        continue
                    skipped_existing = len(selected_solvers) - len(solvers_to_run)
                    if skipped_existing:
                        total_skipped += skipped_existing

                    feasible_solvers, memory_skipped = _memory_feasible_solvers(
                        split,
                        m,
                        args,
                        feature_dtype,
                        solvers_to_run,
                    )
                    if memory_skipped:
                        total_memory_skipped += len(memory_skipped)
                        for solver, peak_gib in memory_skipped:
                            print(
                                f"  SKIP {solver:<18} lam={lam:.0e} m={m} seed={seed}: "
                                f"estimated peak {peak_gib:.3g} GiB exceeds "
                                f"--max-peak-gb={args.max_peak_gb:.3g}"
                            )
                    solvers_to_run = feasible_solvers
                    if not solvers_to_run:
                        continue
                    if skipped_existing or memory_skipped:
                        missing = ", ".join(solvers_to_run)
                        print(
                            f"  RESUME lam={lam:.0e} m={m} seed={seed}: running {missing}"
                        )
                    rows = None
                    try:
                        rows = run_config(
                            split,
                            lam,
                            m,
                            seed,
                            n_stages,
                            inner,
                            run_id=run_id,
                            dataset=ds,
                            transform_chunk_mb=args.transform_chunk_mb,
                            feature_dtype=feature_dtype,
                            solvers=solvers_to_run,
                        )
                        append_rows(out_path, rows)
                        total_rows += len(rows)
                        for row in rows:
                            key = _row_entry_key(row)
                            if key is not None:
                                completed_entries.add(key)
                        for r in rows:
                            print(
                                f"  {r['solver']:<18} lam={lam:.0e} m={m} seed={seed}: "
                                f"acc={r['test_acc']:.4f} "
                                f"solver_train={r['train_time_solver'] or 0:.2f}s "
                                f"total={r['total_time']:.2f}s"
                            )
                    except Exception as e:
                        print(f"  FAILED lam={lam:.0e} m={m} seed={seed}: {e}")
                    finally:
                        rows = None
                        _release_unused_memory()
        split = None
        _release_unused_memory()

    save_metadata(out_path.parent, run_id, [sys.executable] + sys.argv)
    print(
        f"\nWrote {total_rows} rows to {out_path}; skipped {total_skipped} existing rows; "
        f"memory-skipped {total_memory_skipped} solver rows"
    )


if __name__ == "__main__":
    main()
