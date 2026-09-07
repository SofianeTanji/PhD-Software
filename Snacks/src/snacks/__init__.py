"""SNACKS: Scalable Nyström Approximation for Kernel Classification with ASSG."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .kernels import linear_kernel, median_bandwidth, rbf_kernel
from .nystrom import NystromTransformer, nystrom_embedding, select_columns
from .solver import DEFAULT_M_INNER, SolverResult, assg_c, assg_r, rassg_r


class SnacksSVM(ClassifierMixin, BaseEstimator):
    """Kernel SVM via Nyström approximation and RASSG-r optimization.

    Fits a primal linear SVM in the Nyström feature space:
        min_u  F(u) = (1/n) Σ max(0, 1 − y_i z_i^T u) + (λ/2) ‖u‖²

    where z_i = Φ(x_i) ≈ K_{x_i, I} U_r Σ_r^{-1/2} (Nyström embedding).

    The defaults for the restarted, regularized accelerated stochastic
    subgradient schedule match the benchmark configuration.

    Parameters
    ----------
    lam : regularization strength (larger = more regularization)
    m : number of Nyström column points
    kernel : "rbf", "linear", or callable(X, Y) -> kernel matrix
    gamma : RBF bandwidth; "median" uses the median heuristic, float sets it directly
    ridge_mu : ridge added to K_{I,I} before eigendecomposition (numerical stability)
    restarts : number of outer RASSG-r restarts
    stages_per_restart : ASSG-r stages per restart
    m_inner0 : inner iterations in the first restart (grows by `growth`)
    growth : geometric growth factor for m_inner across restarts
    beta0_scale : sets beta = beta0_scale / (lam * sqrt(m_inner)) at each restart
    beta_decay : per-stage geometric decay of beta within a restart
    val_ratio : fraction of training data held out for best-stage selection
    random_state : int or None
    """

    def __init__(
        self,
        lam: float = 1e-3,
        m: int = 100,
        kernel: str | object = "rbf",
        gamma: float | str = "median",
        ridge_mu: float = 1e-6,
        restarts: int = 4,
        stages_per_restart: int = 10,
        m_inner0: int = 512,
        growth: float = 2.0,
        beta0_scale: float = 10.0,
        beta_decay: float = 1.35,
        val_ratio: float = 0.15,
        random_state: int | None = None,
    ):
        self.lam = lam
        self.m = m
        self.kernel = kernel
        self.gamma = gamma
        self.ridge_mu = ridge_mu
        self.restarts = restarts
        self.stages_per_restart = stages_per_restart
        self.m_inner0 = m_inner0
        self.growth = growth
        self.beta0_scale = beta0_scale
        self.beta_decay = beta_decay
        self.val_ratio = val_ratio
        self.random_state = random_state

    def _resolve_kernel(self, X: np.ndarray):
        if callable(self.kernel) and not isinstance(self.kernel, str):
            return self.kernel
        if self.kernel == "linear":
            return linear_kernel
        if self.kernel == "rbf":
            gamma = self.gamma
            if gamma == "median":
                gamma = median_bandwidth(
                    X, rng=np.random.default_rng(self.random_state or 0)
                )
            g = float(gamma)
            return lambda A, B: rbf_kernel(A, B, g)
        raise ValueError(f"Unknown kernel: {self.kernel!r}")

    def _encode_labels(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"Expected 2 classes, got {len(classes)}.")
        y_enc = np.where(y == classes[1], 1.0, -1.0).astype(np.float32)
        return y_enc, classes

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SnacksSVM":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)

        y_enc, self.classes_ = self._encode_labels(y)

        # Validation split for early stopping
        n = X.shape[0]
        n_val = max(1, int(n * self.val_ratio)) if self.val_ratio > 0 else 0
        if n_val > 0:
            val_idx = rng.choice(n, size=n_val, replace=False)
            train_mask = np.ones(n, dtype=bool)
            train_mask[val_idx] = False
            X_tr, y_tr = X[train_mask], y_enc[train_mask]
            X_val, y_val = X[val_idx], y_enc[val_idx]
        else:
            X_tr, y_tr = X, y_enc
            X_val = y_val = None

        kernel_fn = self._resolve_kernel(X_tr)

        # Nyström columns from training data
        m = min(self.m, X_tr.shape[0])
        self.columns_ = select_columns(X_tr, m, rng)
        self.nystrom_ = NystromTransformer(
            kernel_fn,
            self.columns_,
            self.ridge_mu,
            output_dtype=np.float32,
        ).fit()
        Z = self.nystrom_.transform(X_tr)
        Z_val = self.nystrom_.transform(X_val) if X_val is not None else None

        result = rassg_r(
            Z=Z,
            y=y_tr,
            lam=self.lam,
            restarts=self.restarts,
            stages_per_restart=self.stages_per_restart,
            m_inner0=self.m_inner0,
            growth=self.growth,
            beta0_scale=self.beta0_scale,
            beta_decay=self.beta_decay,
            Z_val=Z_val,
            y_val=y_val,
            rng=rng,
        )

        self.u_ = result.u
        self.history_ = result.history
        self.stop_reason_ = result.stop_reason
        self.kernel_fn_ = kernel_fn
        return self

    def _embed(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return self.nystrom_.transform(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self._embed(X) @ self.u_

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return np.where(scores >= 0, self.classes_[1], self.classes_[0])
