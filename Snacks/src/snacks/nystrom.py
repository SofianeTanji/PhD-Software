"""Nystrom approximation: Z = K_{X,I} U_r Sigma_r^{-1/2}.

Optimizations over the baseline:
  * rank truncation (`rank` and/or `eig_rel_tol`): keep only the top
    eigen-directions instead of *all* positive eigenvalues. This is the single
    biggest speed lever because transform cost is O(n * m * r), and it also
    removes the noise blow-up from dividing by near-zero eigenvalues.
  * scaling folded into the projection matrix once (W = U_r / sqrt(sigma_r)),
    so transform is a single matmul with no per-call elementwise scaling.
  * column squared-norms (Y_sq) precomputed once and reused every transform.
  * blocked transform so the n x m kernel block never has to be materialized in
    full (the full matrix is tens of GB for multi-million-row datasets).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


class NystromTransformer:
    def __init__(
        self,
        kernel_fn: Callable[..., np.ndarray],
        columns: np.ndarray,
        ridge_mu: float = 1e-6,
        rank: Optional[int] = None,
        eig_rel_tol: float = 0.0,
        block_size: int = 16384,
        output_dtype=None,
    ):
        if output_dtype is not None and np.dtype(output_dtype) != np.dtype(np.float32):
            raise ValueError("NystromTransformer is float32-only.")
        self.kernel_fn = kernel_fn
        self.columns = np.asarray(columns, dtype=np.float32)
        self.ridge_mu = ridge_mu
        self.rank = rank
        self.eig_rel_tol = eig_rel_tol
        self.block_size = block_size
        self.output_dtype = np.float32
        # cache column squared-norms for the rbf fast path (ignored by kernels
        # whose signature does not accept Y_sq)
        self._Y_sq = np.einsum("ij,ij->i", self.columns, self.columns)[None, :].astype(np.float32)

    def _kernel(self, X: np.ndarray) -> np.ndarray:
        try:
            return np.asarray(self.kernel_fn(X, self.columns, Y_sq=self._Y_sq), dtype=np.float32)
        except TypeError:
            return np.asarray(self.kernel_fn(X, self.columns), dtype=np.float32)

    def fit(self) -> "NystromTransformer":
        m = self.columns.shape[0]
        dtype = np.dtype(np.float32)
        try:
            K_II = np.asarray(self.kernel_fn(self.columns, self.columns, Y_sq=self._Y_sq), dtype=dtype)
        except TypeError:
            K_II = np.asarray(self.kernel_fn(self.columns, self.columns), dtype=dtype)
        K_II += dtype.type(self.ridge_mu) * np.eye(m, dtype=dtype)
        sigma, U = np.linalg.eigh(K_II)
        sigma = sigma[::-1]
        U = U[:, ::-1]

        keep = sigma > 0.0
        if self.eig_rel_tol > 0.0 and np.any(keep):
            keep &= sigma > (sigma[0] * dtype.type(self.eig_rel_tol))
        if not np.any(keep):
            raise ValueError(
                "All eigenvalues of K_II are non-positive after regularization. Increase ridge_mu."
            )
        sigma = sigma[keep]
        U = U[:, keep]
        if self.rank is not None:
            r = min(self.rank, sigma.shape[0])
            sigma = sigma[:r]
            U = U[:, :r]

        self.sigma_r_ = sigma
        self.U_r_ = U.astype(dtype, copy=False)
        self.inv_sqrt_sigma_r_ = (dtype.type(1.0) / np.sqrt(self.sigma_r_)).astype(dtype, copy=False)
        # fold the 1/sqrt(sigma) scaling into the projection matrix once
        self.W_ = (self.U_r_ * self.inv_sqrt_sigma_r_).astype(dtype, copy=False)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        r = self.W_.shape[1]
        bs = self.block_size if self.block_size and self.block_size > 0 else n
        if n <= bs:
            return (self._kernel(X) @ self.W_).astype(np.float32, copy=False)
        out = np.empty((n, r), dtype=np.float32)
        for i in range(0, n, bs):
            out[i:i + bs] = self._kernel(X[i:i + bs]) @ self.W_
        return out


def nystrom_embedding(
    X: np.ndarray,
    kernel_fn: Callable[..., np.ndarray],
    columns: np.ndarray,
    ridge_mu: float = 1e-6,
    rank: Optional[int] = None,
    eig_rel_tol: float = 0.0,
) -> np.ndarray:
    """Convenience wrapper: fit transformer and embed X in one call."""
    return NystromTransformer(
        kernel_fn, columns, ridge_mu, rank=rank, eig_rel_tol=eig_rel_tol
    ).fit().transform(X)


def select_columns(X: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly sample m column rows from X without replacement."""
    idx = rng.choice(X.shape[0], size=m, replace=False)
    return X[idx]