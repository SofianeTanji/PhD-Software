"""Shared helpers for benchmark scripts: canonical CSV schema and metadata."""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ASSG_DEFAULT_INNER_STEPS = 128
RASSG_R_DEFAULT_RESTARTS = 4
RASSG_R_DEFAULT_STAGES_PER_RESTART = 10
RASSG_R_DEFAULT_M_INNER0 = 512
RASSG_R_DEFAULT_GROWTH = 2.0
RASSG_R_DEFAULT_BETA0_SCALE = 10.0
RASSG_R_DEFAULT_BETA_DECAY = 1.35
NYSTROM_DEFAULT_DELTA = 0.05
NYSTROM_DEFAULT_EIGEN_DECAY_P = 0.5
NYSTROM_DEFAULT_MAX_FEATURE_MB = 1024.0
NYSTROM_DEFAULT_MIN_M = 50
NYSTROM_DEFAULT_GRID_FACTORS = (0.25, 0.5, 1.0)

# Canonical raw-result schema. Empty string means "not applicable to this row".
FIELDS = [
    "run_id",
    "script_name",
    "dataset",
    "solver",
    "split_seed",
    "solver_seed",
    "lam",
    "m",
    "gamma",
    "ridge_mu",
    "n_train",
    "n_val",
    "n_test",
    "n_features",
    "is_sparse",
    "n_stages",
    "m_inner",
    "alpha",
    "q",
    "adaptive_decay",
    "val_acc",
    "train_acc",
    "test_acc",
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


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_") + os.urandom(2).hex()


def empty_row(**kwargs) -> dict:
    row = {f: "" for f in FIELDS}
    row.update(kwargs)
    return row


def default_assg_m_inner(
    n_train: int,
    *,
    inner_steps: int = ASSG_DEFAULT_INNER_STEPS,
) -> int:
    """Default ASSG-r inner-loop budget for experiments.

    The tuned default uses a fixed short stage length. Indices are sampled with
    replacement, so this does not need to scale down on tiny datasets.
    """
    if n_train <= 0:
        raise ValueError("n_train must be positive.")
    if inner_steps <= 0:
        raise ValueError("inner_steps must be positive.")

    return inner_steps


def default_nystrom_m_grid(
    n_train: int,
    *,
    delta: float = NYSTROM_DEFAULT_DELTA,
    eigen_decay_p: float = NYSTROM_DEFAULT_EIGEN_DECAY_P,
    max_feature_mb: float | None = NYSTROM_DEFAULT_MAX_FEATURE_MB,
    min_m: int = NYSTROM_DEFAULT_MIN_M,
    factors: tuple[float, ...] = NYSTROM_DEFAULT_GRID_FACTORS,
) -> list[int]:
    """Return a transparent default grid for the Nyström rank m.

    The scale follows the leverage-score result in Della Vecchia et al.:
    under polynomial eigendecay with exponent p, the useful subspace size is
    m roughly n^p log(n / delta). This repository still samples columns
    uniformly, so the helper is a theory-inspired sweep scale, not a formal
    guarantee. The dense feature matrix Z has size n_train x m, so the target
    is capped by a simple memory budget.
    """
    if n_train <= 0:
        raise ValueError("n_train must be positive.")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1).")
    if not 0 < eigen_decay_p < 1:
        raise ValueError("eigen_decay_p must be in (0, 1).")
    if min_m <= 0:
        raise ValueError("min_m must be positive.")
    if not factors:
        raise ValueError("factors must not be empty.")
    if any(f <= 0 for f in factors):
        raise ValueError("all factors must be positive.")
    if max_feature_mb is not None and max_feature_mb <= 0:
        raise ValueError("max_feature_mb must be positive when provided.")

    if max_feature_mb is None:
        memory_cap = n_train
    else:
        memory_cap = int(max_feature_mb * 1024 * 1024 // (8 * n_train))

    cap = max(1, min(n_train, memory_cap))
    log_term = max(1.0, math.log(n_train / delta))
    target = math.ceil((n_train**eigen_decay_p) * log_term)
    target = max(1, min(cap, target))
    lower = min(cap, min_m)

    grid = []
    for factor in factors:
        m = math.ceil(target * factor)
        if target >= lower:
            m = max(lower, m)
        grid.append(max(1, min(cap, m)))
    grid.append(target)
    return sorted(set(grid))


def parse_int_grid(value: str) -> list[int]:
    """Parse a comma-separated positive integer grid."""
    grid = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        item = int(part)
        if item <= 0:
            raise ValueError("grid values must be positive.")
        grid.append(item)
    if not grid:
        raise ValueError("grid must contain at least one value.")
    return sorted(set(grid))


def append_rows(out_path: Path, rows: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return ""


def _pkg_version(name: str) -> str:
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception:
        return ""


def save_metadata(metadata_dir: Path, run_id: str, command: list[str]) -> None:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "command": " ".join(command),
        "python": sys.version,
        "numpy": _pkg_version("numpy"),
        "scikit_learn": _pkg_version("scikit-learn"),
        "numba": _pkg_version("numba"),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "timestamp": datetime.now().isoformat(),
        "notes": "Solver-only timings exclude warmup; RASSG-r optimizer timing excludes validation selection.",
    }
    with open(metadata_dir / f"run_metadata_{run_id}.json", "w") as f:
        json.dump(meta, f, indent=2)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
