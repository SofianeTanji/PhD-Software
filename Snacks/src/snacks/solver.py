"""ASSG-c / ASSG-r: Adaptive Stochastic Subgradient Solvers.

Solves the primal linear SVM objective over Nyström features:

    min_u  F(u) = (1/n) Σ max(0, 1 − y_i z_i^T u) + (λ/2) ‖u‖²

Algorithm:
  - Staged stochastic subgradient descent with warm-start between stages.
  - Each stage runs m_inner SGD steps starting from the previous stage's
    averaged iterate.
  - The stage output is the average of suffix iterates selected by sampling q
    positions with replacement from the last alpha fraction of the stage:
    J_1,...,J_q ~ Unif{⌊(1-alpha)m⌋,...,m}. The implementation stores selected
    positions in a boolean mask, so duplicate positions collapse before
    averaging; q controls the expected number of unique suffix iterates.
  - Step size η = 4/(λ√m) (default) satisfies η·λ <= 1 for m >= 16, keeping the
    regularization shrinkage factor (1−η·λ) nonnegative; η is halved (or
    adaptively decayed) between stages.
  - Inner loop uses scalar scaling: maintain v = s·w so the per-step
    shrinkage v *= (1−η·λ) is O(1) on scalar s. When s drops below 1e-9
    (matching sklearn's WeightVector threshold), w is materialized in-place
    and s reset to 1. The same reset happens once at the end of each stage.
  - Return the iterate with the best validation accuracy seen across all stages.
  - ASSG-r replaces local projection with a per-stage proximal centering
    penalty 1/(2 beta_k) ||u - c_k||^2.
  - Use Cython whole-loop kernels when the extension is available; RASSG-r runs
    the complete restarted optimizer path in one compiled call and performs
    validation selection after optimizer timing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numba
import numpy as np

try:
    from ._cython_solver import inner_stage as _cython_inner_stage
    from ._cython_solver import (
        inner_stage_regularized as _cython_inner_stage_regularized,
    )
    from ._cython_solver import (
        run_all_stages_fixed_decay as _cython_run_all_stages_fixed_decay,
    )
    from ._cython_solver import (
        run_all_stages_regularized_fixed_decay as _cython_run_all_stages_regularized_fixed_decay,
    )
    from ._cython_solver import (
        run_restarted_regularized_path as _cython_run_restarted_regularized_path,
    )
except ImportError:  # pragma: no cover - exercised when extension is not built
    _cython_inner_stage = None
    _cython_inner_stage_regularized = None
    _cython_run_all_stages_fixed_decay = None
    _cython_run_all_stages_regularized_fixed_decay = None
    _cython_run_restarted_regularized_path = None


@dataclass
class SolverResult:
    u: np.ndarray
    n_stages_run: int
    stop_reason: str
    history: list[dict] = field(default_factory=list)
    optimizer_time: float | None = None
    selection_time: float | None = None


_WSCALE_THRESHOLD = 1e-9
DEFAULT_M_INNER = 128
_SOLVER_DTYPE = np.float32


def _as_solver_matrix(Z: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(Z, dtype=_SOLVER_DTYPE)


def _as_solver_vector(y: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(y, dtype=_SOLVER_DTYPE)


def _as_optional_solver_matrix(Z: np.ndarray | None) -> np.ndarray | None:
    return None if Z is None else _as_solver_matrix(Z)


def _as_optional_solver_vector(y: np.ndarray | None) -> np.ndarray | None:
    return None if y is None else _as_solver_vector(y)


def _make_stage_plans(
    rng: np.random.Generator,
    n_train: int,
    n_stages: int,
    m_inner: int,
    suffix_start: int,
    q: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.empty((n_stages, m_inner), dtype=np.int64)
    accumulate = np.zeros((n_stages, m_inner), dtype=np.bool_)
    for stage in range(n_stages):
        indices[stage] = rng.integers(0, n_train, size=m_inner)
        accumulate[stage, rng.integers(suffix_start, m_inner, size=q)] = True
    return indices, accumulate


def _make_restarted_stage_plans(
    rng: np.random.Generator,
    n_train: int,
    restarts: int,
    stages_per_restart: int,
    m_inner0: int,
    growth: float,
    beta0_scale: np.float32,
    beta_decay: np.float32,
    lam: np.float32,
    alpha: float,
    q: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    total_stages = restarts * stages_per_restart
    stage_m_inner = np.empty(total_stages, dtype=np.int64)
    stage_beta = np.empty(total_stages, dtype=np.float32)
    stage_meta: list[dict] = []

    max_m_inner = 0
    stage = 0
    for restart in range(1, restarts + 1):
        m_inner = max(1, int(round(m_inner0 * (growth ** (restart - 1)))))
        beta = beta0_scale / (lam * np.sqrt(_SOLVER_DTYPE(m_inner)))
        max_m_inner = max(max_m_inner, m_inner)
        for local_stage in range(1, stages_per_restart + 1):
            stage_m_inner[stage] = m_inner
            stage_beta[stage] = beta
            stage_meta.append(
                {
                    "stage": stage + 1,
                    "restart": restart,
                    "local_stage": local_stage,
                    "m_inner": m_inner,
                    "eta": None,
                    "beta": float(beta),
                    "eta_schedule": "2*beta/tau",
                }
            )
            beta /= beta_decay
            stage += 1

    indices = np.empty((total_stages, max_m_inner), dtype=np.int64)
    accumulate = np.zeros((total_stages, max_m_inner), dtype=np.bool_)
    for stage, m_inner in enumerate(stage_m_inner):
        suffix_start = min(
            int(np.floor((1.0 - alpha) * int(m_inner))), int(m_inner) - 1
        )
        indices[stage, :m_inner] = rng.integers(0, n_train, size=int(m_inner))
        accumulate[stage, rng.integers(suffix_start, int(m_inner), size=q)] = True

    return indices, accumulate, stage_m_inner, stage_beta, stage_meta


def _can_use_cython_fast_path(
    Z: np.ndarray,
    y: np.ndarray,
    Z_val: np.ndarray | None,
    y_val: np.ndarray | None,
    Z_test: np.ndarray | None,
    y_test: np.ndarray | None,
    compute_diagnostics: bool,
    local_projection: bool,
) -> bool:
    return (
        _cython_run_all_stages_fixed_decay is not None
        and not local_projection
        and not compute_diagnostics
        and Z_val is None
        and y_val is None
        and Z_test is None
        and y_test is None
        and Z.dtype == _SOLVER_DTYPE
        and y.dtype == _SOLVER_DTYPE
        and Z.flags.c_contiguous
        and y.flags.c_contiguous
    )


def _can_use_cython_inner_stage(Z: np.ndarray, y: np.ndarray) -> bool:
    return (
        _cython_inner_stage is not None
        and Z.dtype == _SOLVER_DTYPE
        and y.dtype == _SOLVER_DTYPE
        and Z.flags.c_contiguous
        and y.flags.c_contiguous
    )


def _can_use_cython_regularized_fast_path(
    Z: np.ndarray,
    y: np.ndarray,
    Z_val: np.ndarray | None,
    y_val: np.ndarray | None,
    Z_test: np.ndarray | None,
    y_test: np.ndarray | None,
    compute_diagnostics: bool,
) -> bool:
    return (
        _cython_run_all_stages_regularized_fixed_decay is not None
        and not compute_diagnostics
        and Z_val is None
        and y_val is None
        and Z_test is None
        and y_test is None
        and Z.dtype == _SOLVER_DTYPE
        and y.dtype == _SOLVER_DTYPE
        and Z.flags.c_contiguous
        and y.flags.c_contiguous
    )


def _can_use_cython_regularized_inner_stage(Z: np.ndarray, y: np.ndarray) -> bool:
    return (
        _cython_inner_stage_regularized is not None
        and Z.dtype == _SOLVER_DTYPE
        and y.dtype == _SOLVER_DTYPE
        and Z.flags.c_contiguous
        and y.flags.c_contiguous
    )


def _can_use_cython_restarted_path(Z: np.ndarray, y: np.ndarray) -> bool:
    return (
        _cython_run_restarted_regularized_path is not None
        and Z.dtype == _SOLVER_DTYPE
        and y.dtype == _SOLVER_DTYPE
        and Z.flags.c_contiguous
        and y.flags.c_contiguous
    )


def _cython_fixed_decay_history(
    n_stages: int, eta0: float, total_stage_time: float
) -> list[dict]:
    if n_stages <= 0:
        return []
    per_stage_time = total_stage_time / n_stages
    history = []
    eta = eta0
    for stage in range(n_stages):
        history.append(
            {
                "stage": stage + 1,
                "eta": eta,
                "stage_time": per_stage_time,
                "val_acc": None,
            }
        )
        eta *= 0.5
    return history


def _cython_regularized_fixed_decay_history(
    n_stages: int,
    beta0: float,
    beta_decay: float,
    total_stage_time: float,
) -> list[dict]:
    if n_stages <= 0:
        return []
    per_stage_time = total_stage_time / n_stages
    history = []
    beta = beta0
    for stage in range(n_stages):
        history.append(
            {
                "stage": stage + 1,
                "eta": None,
                "beta": beta,
                "eta_schedule": "2*beta/tau",
                "stage_time": per_stage_time,
                "val_acc": None,
            }
        )
        beta /= beta_decay
    return history


@numba.njit(cache=True)
def _inner_stage(
    center: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    eta: float,
    lam: float,
    accumulate: np.ndarray,  # bool mask of length m_inner
) -> np.ndarray:
    """Run one ASSG-c stage; return the average of iterates where accumulate[k]=True.

    Maintains v = s · w. Shrinkage v *= (1 − η·λ) is O(1) on s.
    When s < 1e-9, materialize w *= s and reset s = 1 (mirrors sklearn's
    WeightVector.scale threshold). Same reset at end of stage.
    """
    r = center.shape[0]
    w = center.copy()
    s = np.float32(1.0)
    v_sum = np.zeros(r, dtype=np.float32)
    count = 0
    eta_lam = eta * lam
    for k in range(len(indices)):
        i = indices[k]
        s = s * (1.0 - eta_lam)
        if s < 1e-9:
            for j in range(r):
                w[j] *= s
            s = np.float32(1.0)
        Zi = Z[i]
        d = np.float32(0.0)
        for j in range(r):
            d += w[j] * Zi[j]
        if y[i] * s * d < np.float32(1.0):
            coef = eta * y[i] / s
            for j in range(r):
                w[j] += coef * Zi[j]
        if accumulate[k]:
            for j in range(r):
                v_sum[j] += s * w[j]
            count += 1
    # end-of-stage reset, matching sklearn's end-of-fit reset_wscale
    for j in range(r):
        w[j] *= s
    inv = np.float32(1.0) / count
    for j in range(r):
        v_sum[j] *= inv
    return v_sum


@numba.njit(cache=True)
def _inner_stage_regularized(
    center: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    eta: float,
    lam: float,
    beta: float,
    accumulate: np.ndarray,  # bool mask of length m_inner
) -> np.ndarray:
    """Run one ASSG-r stage using the paper update with identity projection."""
    r = center.shape[0]
    w = center.copy()
    u_sum = np.zeros(r, dtype=np.float32)
    count = 0
    for j in range(r):
        w[j] = center[j]

    for k in range(len(indices)):
        i = indices[k]
        Zi = Z[i]
        tau = np.float32(k + 1)

        d = np.float32(0.0)
        for j in range(r):
            d += w[j] * Zi[j]
        active = y[i] * d < np.float32(1.0)

        w_coef = (
            np.float32(1.0)
            - (np.float32(2.0) / tau)
            - ((np.float32(2.0) * beta * lam) / tau)
        )
        center_coef = np.float32(2.0) / tau
        if active:
            hinge_coef = np.float32(2.0) * beta * y[i] / tau
            for j in range(r):
                w[j] = w_coef * w[j] + center_coef * center[j] + hinge_coef * Zi[j]
        else:
            for j in range(r):
                w[j] = w_coef * w[j] + center_coef * center[j]

        if accumulate[k]:
            for j in range(r):
                u_sum[j] += w[j]
            count += 1

    inv = np.float32(1.0) / count
    for j in range(r):
        u_sum[j] *= inv
    return u_sum


@numba.njit(cache=True)
def _inner_stage_regularized_l1(
    center: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    lam: float,
    beta: float,
    accumulate: np.ndarray,  # bool mask of length m_inner
) -> np.ndarray:
    """Run one ASSG-r stage for hinge loss with an L1 subgradient."""
    r = center.shape[0]
    w = center.copy()
    u_sum = np.zeros(r, dtype=np.float32)
    count = 0
    for j in range(r):
        w[j] = center[j]

    for k in range(len(indices)):
        i = indices[k]
        Zi = Z[i]
        tau = np.float32(k + 1)

        d = np.float32(0.0)
        for j in range(r):
            d += w[j] * Zi[j]
        active = y[i] * d < np.float32(1.0)

        w_coef = np.float32(1.0) - (np.float32(2.0) / tau)
        center_coef = np.float32(2.0) / tau
        reg_coef = (np.float32(2.0) * beta * lam) / tau
        if active:
            hinge_coef = np.float32(2.0) * beta * y[i] / tau
            for j in range(r):
                if w[j] > np.float32(0.0):
                    reg_subgrad = np.float32(1.0)
                elif w[j] < np.float32(0.0):
                    reg_subgrad = np.float32(-1.0)
                else:
                    reg_subgrad = np.float32(0.0)
                w[j] = (
                    w_coef * w[j]
                    + center_coef * center[j]
                    - reg_coef * reg_subgrad
                    + hinge_coef * Zi[j]
                )
        else:
            for j in range(r):
                if w[j] > np.float32(0.0):
                    reg_subgrad = np.float32(1.0)
                elif w[j] < np.float32(0.0):
                    reg_subgrad = np.float32(-1.0)
                else:
                    reg_subgrad = np.float32(0.0)
                w[j] = (
                    w_coef * w[j]
                    + center_coef * center[j]
                    - reg_coef * reg_subgrad
                )

        if accumulate[k]:
            for j in range(r):
                u_sum[j] += w[j]
            count += 1

    inv = np.float32(1.0) / count
    for j in range(r):
        u_sum[j] *= inv
    return u_sum


@numba.njit(cache=True)
def _inner_stage_local_projected(
    center: np.ndarray,
    Z: np.ndarray,
    Z_norms: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    eta: float,
    lam: float,
    accumulate: np.ndarray,  # bool mask of length m_inner
    radius: float,
) -> tuple[np.ndarray, float, int, int]:
    """Run one stage with per-update projection onto B(center, radius).

    Maintains the same lazy scaling representation as _inner_stage, but checks
    and applies projection in the actual coordinates u = s * w.
    """
    r = center.shape[0]
    w = center.copy()
    s = np.float32(1.0)
    u_sum = np.zeros(r, dtype=np.float32)
    count = 0
    eta_lam = eta * lam
    shrink = 1.0 - eta_lam
    eta_lam_abs = abs(eta_lam)
    shrink_abs = abs(shrink)
    eta_abs = abs(eta)
    center_norm_sq = np.float32(0.0)
    for j in range(r):
        center_norm_sq += center[j] * center[j]
    center_norm = np.sqrt(center_norm_sq)
    radius_sq = radius * radius
    exact_check_threshold = radius * (np.float32(1.0) - np.float32(1e-6))
    distance_bound = np.float32(0.0)
    max_distance_bound_sq = np.float32(0.0)
    projection_count = 0
    exact_check_count = 0

    for k in range(len(indices)):
        i = indices[k]
        s = s * shrink
        if s < 1e-9:
            for j in range(r):
                w[j] *= s
            s = np.float32(1.0)
        Zi = Z[i]

        d = np.float32(0.0)
        for j in range(r):
            d += w[j] * Zi[j]

        if y[i] * s * d < np.float32(1.0):
            coef = eta * y[i] / s
            for j in range(r):
                w[j] += coef * Zi[j]
            distance_bound = (
                shrink_abs * distance_bound
                + eta_lam_abs * center_norm
                + eta_abs * Z_norms[i]
            )
        else:
            distance_bound = shrink_abs * distance_bound + eta_lam_abs * center_norm

        if distance_bound > exact_check_threshold:
            exact_check_count += 1
            norm_sq = np.float32(0.0)
            for j in range(r):
                diff = s * w[j] - center[j]
                norm_sq += diff * diff
            if norm_sq > radius_sq:
                projection_count += 1
                norm = np.sqrt(norm_sq)
                scale = radius / norm
                for j in range(r):
                    w[j] = center[j] + scale * (s * w[j] - center[j])
                s = np.float32(1.0)
                norm_sq = np.float32(0.0)
                for j in range(r):
                    diff = w[j] - center[j]
                    norm_sq += diff * diff
            distance_bound = np.sqrt(norm_sq)
            distance_bound_sq = norm_sq
        else:
            distance_bound_sq = distance_bound * distance_bound

        if distance_bound_sq > max_distance_bound_sq:
            max_distance_bound_sq = distance_bound_sq

        if accumulate[k]:
            for j in range(r):
                u_sum[j] += s * w[j]
            count += 1

    inv = np.float32(1.0) / count
    for j in range(r):
        u_sum[j] *= inv
    return u_sum, np.sqrt(max_distance_bound_sq), projection_count, exact_check_count


def assg_c(
    Z: np.ndarray,
    y: np.ndarray,
    lam: float,
    n_stages: int,
    m_inner: int = DEFAULT_M_INNER,
    eta0: float | None = None,
    Z_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    Z_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    adaptive_decay: bool = True,
    decay_slow: float = 1.7,
    decay_fast: float = 2.3,
    improvement_threshold: float = 1e-3,
    val_patience: int = 3,
    min_stages: int = 5,
    alpha: float = 0.5,
    q: int = 16,
    rng: np.random.Generator | None = None,
    compute_diagnostics: bool = False,
    local_projection: bool = False,
    local_projection_d0: float | None = None,
) -> SolverResult:
    """Run ASSG-c and return the best iterate by validation accuracy.

    Parameters
    ----------
    Z : (n, r) float32 Nyström feature matrix
    y : (n,) float32 labels in {-1, +1}
    lam : regularization parameter
    n_stages : maximum number of stages
    m_inner : inner iterations per stage; defaults to 128
    eta0 : initial step size; defaults to 4/(lam*sqrt(m_inner))
    Z_val, y_val : optional validation set for early stopping
    adaptive_decay : if True use two-rate decay, else halve each stage
    decay_slow : decay divisor when improvement >= improvement_threshold
    decay_fast : decay divisor when improvement < improvement_threshold
    improvement_threshold : relative improvement threshold for decay-rate selection
    val_patience : stop after this many stages with no val_acc improvement
    min_stages : do not stop early before this many stages
    alpha : suffix fraction; average over iterates from step ⌊(1-alpha)·m⌋ onward
    q : number of iterates sampled from the suffix for the stage average
    rng : random generator for index sampling
    local_projection : if True, project each inner update onto the local ASSG
        ball B(center_k, D_k), where D_1 = 2 max_i ||Z_i|| / lam and
        D_k = D_1 / 2**k for zero-based stage index k.
    local_projection_d0 : optional override for the initial projection radius
        D_1.  When None, use 2 max_i ||Z_i|| / lam.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    Z = _as_solver_matrix(Z)
    y = _as_solver_vector(y)
    Z_val = _as_optional_solver_matrix(Z_val)
    y_val = _as_optional_solver_vector(y_val)
    Z_test = _as_optional_solver_matrix(Z_test)
    y_test = _as_optional_solver_vector(y_test)
    lam = _SOLVER_DTYPE(lam)
    if eta0 is None:
        eta0 = _SOLVER_DTYPE(4.0) / (lam * np.sqrt(_SOLVER_DTYPE(m_inner)))
    else:
        eta0 = _SOLVER_DTYPE(eta0)

    n, r = Z.shape
    center = np.zeros(r, dtype=_SOLVER_DTYPE)
    eta = eta0
    local_projection_d1 = None
    Z_norms = None
    if local_projection:
        if local_projection_d0 is not None and local_projection_d0 <= 0:
            raise ValueError("local_projection_d0 must be positive when provided.")
        Z_norms = np.sqrt(np.sum(Z * Z, axis=1))
        if local_projection_d0 is None:
            max_feature_norm = float(np.max(Z_norms))
            local_projection_d1 = 2.0 * max_feature_norm / lam
        else:
            local_projection_d1 = float(local_projection_d0)

    best_u = center.copy()
    best_val_acc = -np.inf
    stages_no_improve = 0
    prev_val_acc = None
    history = []
    stop_reason = "max_stages"

    suffix_start = min(int(np.floor((1.0 - alpha) * m_inner)), m_inner - 1)

    if _can_use_cython_fast_path(
        Z,
        y,
        Z_val,
        y_val,
        Z_test,
        y_test,
        compute_diagnostics,
        local_projection,
    ):
        indices, acc_masks = _make_stage_plans(
            rng, n, n_stages, m_inner, suffix_start, q
        )
        _t0 = time.perf_counter()
        best_u = _cython_run_all_stages_fixed_decay(
            Z,
            y,
            indices,
            np.ascontiguousarray(acc_masks.view(np.uint8)),
            eta0,
            lam,
        )
        stage_time = time.perf_counter() - _t0
        return SolverResult(
            u=best_u,
            n_stages_run=n_stages,
            stop_reason=stop_reason,
            history=_cython_fixed_decay_history(n_stages, eta0, stage_time),
        )

    for t in range(n_stages):
        indices = rng.integers(0, n, size=m_inner)
        acc_mask = np.zeros(m_inner, dtype=np.bool_)
        acc_mask[rng.integers(suffix_start, m_inner, size=q)] = True
        _t0 = time.perf_counter()
        local_projection_radius = None
        local_projection_max_distance = None
        local_projection_count = None
        local_projection_exact_check_count = None
        if local_projection:
            local_projection_radius = local_projection_d1 / (2.0**t)
            (
                u,
                local_projection_max_distance,
                local_projection_count,
                local_projection_exact_check_count,
            ) = _inner_stage_local_projected(
                center,
                Z,
                Z_norms,
                y,
                indices,
                eta,
                lam,
                acc_mask,
                local_projection_radius,
            )
        elif _can_use_cython_inner_stage(Z, y):
            u = _cython_inner_stage(
                center,
                Z,
                y,
                indices,
                eta,
                lam,
                np.ascontiguousarray(acc_mask.view(np.uint8)),
            )
        else:
            u = _inner_stage(center, Z, y, indices, eta, lam, acc_mask)
        stage_time = time.perf_counter() - _t0

        val_acc = None
        if Z_val is not None and y_val is not None:
            val_acc = float(np.mean((y_val * (Z_val @ u)) > 0))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_u = u.copy()
                stages_no_improve = 0
            else:
                stages_no_improve += 1

        entry: dict = {
            "stage": t + 1,
            "eta": eta,
            "stage_time": stage_time,
            "val_acc": val_acc,
        }
        if local_projection:
            entry["local_projection_radius"] = float(local_projection_radius)
            entry["local_projection_stage_distance"] = float(
                np.sqrt(np.dot(u - center, u - center))
            )
            entry["local_projection_max_distance"] = float(
                local_projection_max_distance
            )
            entry["local_projection_count"] = int(local_projection_count)
            entry["local_projection_exact_check_count"] = int(
                local_projection_exact_check_count
            )
        if compute_diagnostics:
            scores = Z @ u
            hinge = np.maximum(0.0, 1.0 - y * scores)
            hinge_loss = float(np.mean(hinge))
            reg = 0.5 * lam * float(np.dot(u, u))
            entry["train_acc"] = float(np.mean((y * (Z @ best_u)) > 0))
            entry["train_hinge_loss"] = hinge_loss
            entry["regularization_term"] = reg
            entry["train_objective"] = hinge_loss + reg
            entry["active_hinge_fraction"] = float(np.mean(hinge > 0))
            entry["weight_norm"] = float(np.sqrt(np.dot(u, u)))
            if Z_test is not None and y_test is not None:
                entry["test_acc"] = float(np.mean((y_test * (Z_test @ best_u)) > 0))
        history.append(entry)

        if adaptive_decay and prev_val_acc is not None and val_acc is not None:
            improvement = val_acc - prev_val_acc
            decay = decay_slow if improvement >= improvement_threshold else decay_fast
        else:
            decay = 2.0

        eta /= decay
        center = u.copy()
        prev_val_acc = val_acc

        if (
            Z_val is not None
            and (t + 1) >= min_stages
            and stages_no_improve >= val_patience
        ):
            stop_reason = "val_patience"
            break

    if Z_val is None:
        best_u = u.copy()

    return SolverResult(
        u=best_u,
        n_stages_run=len(history),
        stop_reason=stop_reason,
        history=history,
    )


def assg_r(
    Z: np.ndarray,
    y: np.ndarray,
    lam: float,
    n_stages: int,
    m_inner: int = DEFAULT_M_INNER,
    eta0: float | None = None,
    beta0: float | None = None,
    beta_decay: float = 2.0,
    Z_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    Z_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    adaptive_decay: bool = True,
    decay_slow: float = 1.7,
    decay_fast: float = 2.3,
    improvement_threshold: float = 1e-3,
    val_patience: int = 3,
    min_stages: int = 5,
    alpha: float = 0.5,
    q: int = 16,
    rng: np.random.Generator | None = None,
    compute_diagnostics: bool = False,
) -> SolverResult:
    """Run ASSG-r and return the best iterate by validation accuracy.

    This follows the paper update

        u_{tau+1} = (1 - 2/tau) u_tau + (2/tau) center
                    - (2 beta_k/tau) partial f(u_tau; xi_tau)

    with identity projection for the unconstrained domain.  The only deliberate
    departure from the paper pseudocode is that the stage output uses the same
    suffix-sampled averaging policy as ``assg_c`` instead of averaging every
    stage iterate.  ``eta0`` is retained for API compatibility and ignored; the
    paper update uses the per-iteration step scale ``2 beta_k / tau``.
    With validation data, ASSG-r uses the same best-stage selection and
    patience stopping policy as ``assg_c``.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    Z = _as_solver_matrix(Z)
    y = _as_solver_vector(y)
    Z_val = _as_optional_solver_matrix(Z_val)
    y_val = _as_optional_solver_vector(y_val)
    Z_test = _as_optional_solver_matrix(Z_test)
    y_test = _as_optional_solver_vector(y_test)
    lam = _SOLVER_DTYPE(lam)
    beta_decay = _SOLVER_DTYPE(beta_decay)
    if lam <= 0:
        raise ValueError("lam must be positive.")
    if beta0 is None:
        beta0 = _SOLVER_DTYPE(1.0) / (lam * np.sqrt(_SOLVER_DTYPE(m_inner)))
    else:
        beta0 = _SOLVER_DTYPE(beta0)
    if beta0 <= 0:
        raise ValueError("beta0 must be positive.")
    if beta_decay != _SOLVER_DTYPE(2.0):
        raise ValueError("paper-exact ASSG-r uses beta_decay=2.0.")

    n, r = Z.shape
    center = np.zeros(r, dtype=_SOLVER_DTYPE)
    beta = beta0

    best_u = center.copy()
    best_val_acc = -np.inf
    stages_no_improve = 0
    history = []
    stop_reason = "max_stages"

    suffix_start = min(int(np.floor((1.0 - alpha) * m_inner)), m_inner - 1)

    if _can_use_cython_regularized_fast_path(
        Z,
        y,
        Z_val,
        y_val,
        Z_test,
        y_test,
        compute_diagnostics,
    ):
        indices, acc_masks = _make_stage_plans(
            rng, n, n_stages, m_inner, suffix_start, q
        )
        _t0 = time.perf_counter()
        best_u = _cython_run_all_stages_regularized_fixed_decay(
            Z,
            y,
            indices,
            np.ascontiguousarray(acc_masks.view(np.uint8)),
            0.0,
            lam,
            beta0,
            beta_decay,
        )
        stage_time = time.perf_counter() - _t0
        return SolverResult(
            u=best_u,
            n_stages_run=n_stages,
            stop_reason=stop_reason,
            history=_cython_regularized_fixed_decay_history(
                n_stages, beta0, beta_decay, stage_time
            ),
        )

    for t in range(n_stages):
        indices = rng.integers(0, n, size=m_inner)
        acc_mask = np.zeros(m_inner, dtype=np.bool_)
        acc_mask[rng.integers(suffix_start, m_inner, size=q)] = True
        _t0 = time.perf_counter()
        if _can_use_cython_regularized_inner_stage(Z, y):
            u = _cython_inner_stage_regularized(
                center,
                Z,
                y,
                indices,
                0.0,
                lam,
                beta,
                np.ascontiguousarray(acc_mask.view(np.uint8)),
            )
        else:
            u = _inner_stage_regularized(
                center, Z, y, indices, 0.0, lam, beta, acc_mask
            )
        stage_time = time.perf_counter() - _t0

        val_acc = None
        if Z_val is not None and y_val is not None:
            val_acc = float(np.mean((y_val * (Z_val @ u)) > 0))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_u = u.copy()
                stages_no_improve = 0
            else:
                stages_no_improve += 1

        entry: dict = {
            "stage": t + 1,
            "eta": None,
            "beta": beta,
            "eta_schedule": "2*beta/tau",
            "stage_time": stage_time,
            "val_acc": val_acc,
        }
        if compute_diagnostics:
            scores = Z @ u
            hinge = np.maximum(0.0, 1.0 - y * scores)
            hinge_loss = float(np.mean(hinge))
            reg = 0.5 * lam * float(np.dot(u, u))
            entry["train_acc"] = float(np.mean((y * (Z @ u)) > 0))
            entry["train_hinge_loss"] = hinge_loss
            entry["regularization_term"] = reg
            entry["train_objective"] = hinge_loss + reg
            entry["active_hinge_fraction"] = float(np.mean(hinge > 0))
            entry["weight_norm"] = float(np.sqrt(np.dot(u, u)))
            if Z_test is not None and y_test is not None:
                entry["test_acc"] = float(np.mean((y_test * (Z_test @ u)) > 0))
        history.append(entry)

        beta /= beta_decay
        center = u.copy()

        if (
            Z_val is not None
            and (t + 1) >= min_stages
            and stages_no_improve >= val_patience
        ):
            stop_reason = "val_patience"
            break

    if Z_val is None:
        best_u = center.copy()

    return SolverResult(
        u=best_u,
        n_stages_run=len(history),
        stop_reason=stop_reason,
        history=history,
    )


def rassg_r(
    Z: np.ndarray,
    y: np.ndarray,
    lam: float,
    restarts: int = 4,
    stages_per_restart: int = 10,
    m_inner0: int = 512,
    growth: float = 2.0,
    beta0_scale: float = 10.0,
    beta_decay: float = 1.35,
    Z_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    Z_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    alpha: float = 0.5,
    q: int = 16,
    rng: np.random.Generator | None = None,
    compute_diagnostics: bool = False,
) -> SolverResult:
    """Run ASSG-r with outer restarts and return the best validation iterate.

    Each restart warm-starts from the previous restart's last stage, resets
    ``beta = beta0_scale / (lam * sqrt(m_inner))``, and increases
    ``m_inner`` geometrically by ``growth``. Stage selection remains based on
    validation accuracy when validation data are supplied.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    Z = _as_solver_matrix(Z)
    y = _as_solver_vector(y)
    Z_val = _as_optional_solver_matrix(Z_val)
    y_val = _as_optional_solver_vector(y_val)
    Z_test = _as_optional_solver_matrix(Z_test)
    y_test = _as_optional_solver_vector(y_test)
    lam = _SOLVER_DTYPE(lam)
    beta0_scale = _SOLVER_DTYPE(beta0_scale)
    beta_decay = _SOLVER_DTYPE(beta_decay)
    if lam <= 0:
        raise ValueError("lam must be positive.")
    if restarts <= 0:
        raise ValueError("restarts must be positive.")
    if stages_per_restart <= 0:
        raise ValueError("stages_per_restart must be positive.")
    if m_inner0 <= 0:
        raise ValueError("m_inner0 must be positive.")
    if growth <= 0:
        raise ValueError("growth must be positive.")
    if beta0_scale <= 0:
        raise ValueError("beta0_scale must be positive.")
    if beta_decay <= 0:
        raise ValueError("beta_decay must be positive.")

    n, r = Z.shape

    if _can_use_cython_restarted_path(Z, y):
        (
            indices,
            acc_masks,
            stage_m_inner,
            stage_beta,
            stage_meta,
        ) = _make_restarted_stage_plans(
            rng,
            n,
            restarts,
            stages_per_restart,
            m_inner0,
            growth,
            beta0_scale,
            beta_decay,
            lam,
            alpha,
            q,
        )
        _t0 = time.perf_counter()
        path = _cython_run_restarted_regularized_path(
            Z,
            y,
            indices,
            np.ascontiguousarray(acc_masks.view(np.uint8)),
            stage_m_inner,
            stage_beta,
            lam,
        )
        optimizer_time = time.perf_counter() - _t0

        _t_select = time.perf_counter()
        best_u = np.zeros(r, dtype=_SOLVER_DTYPE)
        best_val_acc = -np.inf
        history = []
        stage_time_scale = optimizer_time / max(1, int(np.sum(stage_m_inner)))

        for idx in range(path.shape[0]):
            u = path[idx]
            val_acc = None
            if Z_val is not None and y_val is not None:
                val_acc = float(np.mean((y_val * (Z_val @ u)) > 0))
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_u = u.copy()

            entry = dict(stage_meta[idx])
            entry["stage_time"] = float(stage_m_inner[idx]) * stage_time_scale
            entry["val_acc"] = val_acc
            if compute_diagnostics:
                scores = Z @ u
                hinge = np.maximum(np.float32(0.0), np.float32(1.0) - y * scores)
                hinge_loss = float(np.mean(hinge))
                reg = float(np.float32(0.5) * lam * np.dot(u, u))
                entry["train_acc"] = float(np.mean((y * scores) > 0))
                entry["train_hinge_loss"] = hinge_loss
                entry["regularization_term"] = reg
                entry["train_objective"] = hinge_loss + reg
                entry["active_hinge_fraction"] = float(np.mean(hinge > 0))
                entry["weight_norm"] = float(np.sqrt(np.dot(u, u)))
                if Z_test is not None and y_test is not None:
                    entry["test_acc"] = float(np.mean((y_test * (Z_test @ u)) > 0))
            history.append(entry)

        if Z_val is None:
            best_u = path[-1].copy()

        selection_time = time.perf_counter() - _t_select
        return SolverResult(
            u=best_u,
            n_stages_run=path.shape[0],
            stop_reason="max_restarts",
            history=history,
            optimizer_time=optimizer_time,
            selection_time=selection_time,
        )

    center = np.zeros(r, dtype=_SOLVER_DTYPE)
    best_u = center.copy()
    best_val_acc = -np.inf
    history = []
    global_stage = 0
    optimizer_time = 0.0
    selection_time = 0.0

    for restart in range(1, restarts + 1):
        m_inner = max(1, int(round(m_inner0 * (growth ** (restart - 1)))))
        beta = beta0_scale / (lam * np.sqrt(_SOLVER_DTYPE(m_inner)))
        suffix_start = min(int(np.floor((1.0 - alpha) * m_inner)), m_inner - 1)

        for local_stage in range(1, stages_per_restart + 1):
            global_stage += 1
            indices = rng.integers(0, n, size=m_inner)
            acc_mask = np.zeros(m_inner, dtype=np.bool_)
            acc_mask[rng.integers(suffix_start, m_inner, size=q)] = True

            _t0 = time.perf_counter()
            if _can_use_cython_regularized_inner_stage(Z, y):
                u = _cython_inner_stage_regularized(
                    center,
                    Z,
                    y,
                    indices,
                    0.0,
                    lam,
                    beta,
                    np.ascontiguousarray(acc_mask.view(np.uint8)),
                )
            else:
                u = _inner_stage_regularized(
                    center, Z, y, indices, 0.0, lam, beta, acc_mask
                )
            stage_time = time.perf_counter() - _t0
            optimizer_time += stage_time

            val_acc = None
            _t_select = time.perf_counter()
            if Z_val is not None and y_val is not None:
                val_acc = float(np.mean((y_val * (Z_val @ u)) > 0))
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_u = u.copy()

            entry: dict = {
                "stage": global_stage,
                "restart": restart,
                "local_stage": local_stage,
                "m_inner": m_inner,
                "eta": None,
                "beta": beta,
                "eta_schedule": "2*beta/tau",
                "stage_time": stage_time,
                "val_acc": val_acc,
            }
            if compute_diagnostics:
                scores = Z @ u
                hinge = np.maximum(0.0, 1.0 - y * scores)
                hinge_loss = float(np.mean(hinge))
                reg = 0.5 * lam * float(np.dot(u, u))
                entry["train_acc"] = float(np.mean((y * scores) > 0))
                entry["train_hinge_loss"] = hinge_loss
                entry["regularization_term"] = reg
                entry["train_objective"] = hinge_loss + reg
                entry["active_hinge_fraction"] = float(np.mean(hinge > 0))
                entry["weight_norm"] = float(np.sqrt(np.dot(u, u)))
                if Z_test is not None and y_test is not None:
                    entry["test_acc"] = float(np.mean((y_test * (Z_test @ u)) > 0))
            selection_time += time.perf_counter() - _t_select
            history.append(entry)

            center = u.copy()
            beta /= beta_decay

    if Z_val is None:
        best_u = center.copy()

    return SolverResult(
        u=best_u,
        n_stages_run=len(history),
        stop_reason="max_restarts",
        history=history,
        optimizer_time=optimizer_time,
        selection_time=selection_time,
    )


def rassg_r_l1(
    Z: np.ndarray,
    y: np.ndarray,
    lam: float,
    restarts: int = 4,
    stages_per_restart: int = 10,
    m_inner0: int = 512,
    growth: float = 2.0,
    beta0_scale: float = 10.0,
    beta_decay: float = 1.35,
    Z_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    Z_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    alpha: float = 0.5,
    q: int = 16,
    rng: np.random.Generator | None = None,
    compute_diagnostics: bool = False,
) -> SolverResult:
    """Run restarted ASSG-r for hinge loss with an L1 regularizer.

    This is the L1 counterpart of :func:`rassg_r`: it uses the same Nyström
    feature matrix, restart schedule, suffix averaging, and validation-stage
    selection, but replaces the L2 oracle term ``lambda * u`` by
    ``lambda * sign(u)`` with ``sign(0) = 0``.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    Z = _as_solver_matrix(Z)
    y = _as_solver_vector(y)
    Z_val = _as_optional_solver_matrix(Z_val)
    y_val = _as_optional_solver_vector(y_val)
    Z_test = _as_optional_solver_matrix(Z_test)
    y_test = _as_optional_solver_vector(y_test)
    lam = _SOLVER_DTYPE(lam)
    beta0_scale = _SOLVER_DTYPE(beta0_scale)
    beta_decay = _SOLVER_DTYPE(beta_decay)
    if lam <= 0:
        raise ValueError("lam must be positive.")
    if restarts <= 0:
        raise ValueError("restarts must be positive.")
    if stages_per_restart <= 0:
        raise ValueError("stages_per_restart must be positive.")
    if m_inner0 <= 0:
        raise ValueError("m_inner0 must be positive.")
    if growth <= 0:
        raise ValueError("growth must be positive.")
    if beta0_scale <= 0:
        raise ValueError("beta0_scale must be positive.")
    if beta_decay <= 0:
        raise ValueError("beta_decay must be positive.")

    n, r = Z.shape
    center = np.zeros(r, dtype=_SOLVER_DTYPE)
    best_u = center.copy()
    best_val_acc = -np.inf
    history = []
    global_stage = 0
    optimizer_time = 0.0
    selection_time = 0.0

    for restart in range(1, restarts + 1):
        m_inner = max(1, int(round(m_inner0 * (growth ** (restart - 1)))))
        beta = beta0_scale / (lam * np.sqrt(_SOLVER_DTYPE(m_inner)))
        suffix_start = min(int(np.floor((1.0 - alpha) * m_inner)), m_inner - 1)

        for local_stage in range(1, stages_per_restart + 1):
            global_stage += 1
            indices = rng.integers(0, n, size=m_inner)
            acc_mask = np.zeros(m_inner, dtype=np.bool_)
            acc_mask[rng.integers(suffix_start, m_inner, size=q)] = True

            _t0 = time.perf_counter()
            u = _inner_stage_regularized_l1(
                center, Z, y, indices, lam, beta, acc_mask
            )
            stage_time = time.perf_counter() - _t0
            optimizer_time += stage_time

            val_acc = None
            _t_select = time.perf_counter()
            if Z_val is not None and y_val is not None:
                val_acc = float(np.mean((y_val * (Z_val @ u)) > 0))
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_u = u.copy()

            entry: dict = {
                "stage": global_stage,
                "restart": restart,
                "local_stage": local_stage,
                "m_inner": m_inner,
                "eta": None,
                "beta": beta,
                "eta_schedule": "2*beta/tau",
                "stage_time": stage_time,
                "val_acc": val_acc,
            }
            if compute_diagnostics:
                scores = Z @ u
                hinge = np.maximum(0.0, 1.0 - y * scores)
                hinge_loss = float(np.mean(hinge))
                reg = lam * float(np.sum(np.abs(u)))
                entry["train_acc"] = float(np.mean((y * scores) > 0))
                entry["train_hinge_loss"] = hinge_loss
                entry["regularization_term"] = reg
                entry["train_objective"] = hinge_loss + reg
                entry["active_hinge_fraction"] = float(np.mean(hinge > 0))
                entry["weight_l1_norm"] = float(np.sum(np.abs(u)))
                entry["weight_nnz"] = int(np.count_nonzero(u))
                if Z_test is not None and y_test is not None:
                    entry["test_acc"] = float(np.mean((y_test * (Z_test @ u)) > 0))
            selection_time += time.perf_counter() - _t_select
            history.append(entry)

            center = u.copy()
            beta /= beta_decay

    if Z_val is None:
        best_u = center.copy()

    return SolverResult(
        u=best_u,
        n_stages_run=len(history),
        stop_reason="max_restarts",
        history=history,
        optimizer_time=optimizer_time,
        selection_time=selection_time,
    )
