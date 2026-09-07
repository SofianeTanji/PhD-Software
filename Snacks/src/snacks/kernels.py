"""Kernel functions: linear, RBF, and median bandwidth heuristic.

Optimized RBF:
  * squared norms via einsum (no n*d temporary copy)
  * fused exp via numexpr when more than one core is available (multithreaded,
    no large intermediate). Falls back to plain numpy.exp on a single core,
    where numexpr only adds overhead.
"""

from __future__ import annotations

import numpy as np

try:
    import numexpr as _ne
    _NE_THREADS = _ne.detect_number_of_cores()
    _USE_NE = _NE_THREADS > 1
    if _USE_NE:
        _ne.set_num_threads(_NE_THREADS)
except Exception:  # pragma: no cover
    _ne = None
    _USE_NE = False


def linear_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    return X @ Y.T


def rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float, Y_sq: np.ndarray | None = None) -> np.ndarray:
    """K(x, y) = exp(-gamma * ||x - y||^2).

    Y_sq: optional precomputed row squared-norms of Y, shape (1, m). When the
    same Y (columns) is reused across many transform calls, passing this in
    avoids recomputing it each time.
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    gamma = np.float32(gamma)
    X_sq = np.einsum("ij,ij->i", X, X)[:, None]
    if Y_sq is None:
        Y_sq = np.einsum("ij,ij->i", Y, Y)[None, :]
    XY = X @ Y.T  # BLAS (multithreaded)
    if _USE_NE:
        d = X_sq + Y_sq - np.float32(2.0) * XY
        return _ne.evaluate("exp(-gamma * where(d > 0.0, d, 0.0))")
    sq = np.maximum(X_sq + Y_sq - np.float32(2.0) * XY, np.float32(0.0))
    return np.exp(-gamma * sq).astype(np.float32, copy=False)


def median_bandwidth(X: np.ndarray, n_subsample: int = 2000, rng: np.random.Generator | None = None) -> float:
    """Return gamma = 1 / median(pairwise squared distances) estimated on a subsample."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = X.shape[0]
    if n > n_subsample:
        idx = rng.choice(n, n_subsample, replace=False)
        X = X[idx]
    X = np.asarray(X, dtype=np.float32)
    sq = np.sum(X ** 2, axis=1)
    sq_dists = sq[:, None] + sq[None, :] - np.float32(2.0) * (X @ X.T)
    np.maximum(sq_dists, np.float32(0.0), out=sq_dists)
    upper = sq_dists[np.triu_indices_from(sq_dists, k=1)]
    med = float(np.median(upper))
    if med <= 0.0:
        nz = upper[upper > 0]
        med = float(np.mean(nz)) if len(nz) > 0 else 1.0
    return 1.0 / med