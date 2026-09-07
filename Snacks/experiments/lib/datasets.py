"""Minimal dataset loading for SNACKS paper experiments.

Datasets are expected in LIBSVM format. Download instructions: experiments/README.md.

Split protocol:
- Datasets with an official test file: test = official, train/val = 80/20 stratified.
- Datasets without an official test file: 60/20/20 stratified split.

All dense features are standardized (fit on train, applied to val/test).
Sparse matrices are kept sparse.
Labels are mapped to {-1, +1}.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EXPERIMENTS_DIR / "data"


@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_train: int
    n_features: int
    is_sparse: bool

    @property
    def name(self) -> str:
        return self._name


def _load_libsvm(
    name: str,
    n_features: int | None = None,
    dtype=np.float32,
):
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "See experiments/README.md for download instructions."
        )
    return load_svmlight_file(str(path), n_features=n_features, dtype=dtype)


def _to_pm1(y: np.ndarray) -> np.ndarray:
    classes = np.unique(y)
    assert len(classes) == 2, f"Expected binary labels, got {classes}"
    lo, hi = classes
    return np.where(y == hi, 1.0, -1.0).astype(np.float32)


def _pad_to_width(X, width: int):
    if X.shape[1] == width:
        return X
    if sp.issparse(X):
        extra = sp.csr_matrix((X.shape[0], width - X.shape[1]), dtype=X.dtype)
        return sp.hstack([X, extra], format="csr")
    return np.pad(X, ((0, 0), (0, width - X.shape[1])))


def _limit_features(X_train, X_val, X_test, max_features: int | None):
    if max_features is None or X_train.shape[1] <= max_features:
        return X_train, X_val, X_test
    if max_features <= 0:
        raise ValueError("max_features must be positive.")

    if sp.issparse(X_train):
        counts = np.asarray(X_train.getnnz(axis=0)).ravel()
    else:
        counts = np.count_nonzero(X_train, axis=0)

    keep = np.argpartition(counts, -max_features)[-max_features:]
    keep = keep[np.lexsort((keep, -counts[keep]))]
    return X_train[:, keep], X_val[:, keep], X_test[:, keep]


_REGISTRY: dict[str, dict] = {
    "mushrooms": {},
    "a1a": {"test": "a1a_test"},
    "splice": {"test": "splice_test"},
    "w8a": {"test": "w8a_test"},
    "ijcnn1": {"test": "ijcnn1_test"},
    "madelon": {"test": "madelon_test"},
    "mnist": {"binarize": lambda y: np.where(y < 5, -1.0, 1.0).astype(np.float32)},
    "covtype.binary": {},
    "epsilon": {"test": "epsilon_test"},
    "SUSY": {"tail_test": 500_000},
    "HIGGS": {"tail_test": 500_000},
    "YearPredictionMSD": {
        "test": "YearPredictionMSD_test",
        "binarize": lambda y: np.where(y >= 2000, 1.0, -1.0).astype(np.float32),
    },
    "news20binary": {"file": "news20.binary.bz2", "max_features": 200_000},
}


def load_dataset(name: str, seed: int = 42, dtype=np.float32) -> Split:
    """Load a dataset by name and return train/val/test splits."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset: {name!r}. Available: {list(_REGISTRY)}")
    cfg = _REGISTRY[name]

    file_name = cfg.get("file", name)
    binarize = cfg.get("binarize")
    X, y = _load_libsvm(file_name, dtype=dtype)
    y = (binarize(y) if binarize else _to_pm1(y)).astype(np.float32, copy=False)

    if "test" in cfg:
        X_test, y_test = _load_libsvm(cfg["test"], dtype=dtype)
        width = max(X.shape[1], X_test.shape[1])
        X = _pad_to_width(X, width)
        X_test = _pad_to_width(X_test, width)
        y_test = (binarize(y_test) if binarize else _to_pm1(y_test)).astype(
            np.float32, copy=False
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
    elif "tail_test" in cfg:
        n_test = int(cfg["tail_test"])
        if X.shape[0] <= n_test:
            raise ValueError(
                f"{name} has only {X.shape[0]} rows; cannot reserve {n_test} for test"
            )
        X_trainval, X_test = X[:-n_test], X[-n_test:]
        y_trainval, y_test = y[:-n_test], y[-n_test:]
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval,
            y_trainval,
            test_size=0.2,
            random_state=seed,
            stratify=y_trainval,
        )
    else:
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, test_size=0.4, random_state=seed, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp
        )

    X_train, X_val, X_test = _limit_features(
        X_train,
        X_val,
        X_test,
        cfg.get("max_features"),
    )

    is_sparse = sp.issparse(X_train)
    if not is_sparse:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(np.asarray(X_train, dtype=dtype)).astype(
            dtype, copy=False
        )
        X_val = scaler.transform(np.asarray(X_val, dtype=dtype)).astype(
            dtype, copy=False
        )
        X_test = scaler.transform(np.asarray(X_test, dtype=dtype)).astype(
            dtype, copy=False
        )
    else:
        X_train = X_train.astype(dtype)
        X_val = X_val.astype(dtype)
        X_test = X_test.astype(dtype)

    return Split(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        n_train=X_train.shape[0],
        n_features=X_train.shape[1] if not is_sparse else X_train.shape[1],
        is_sparse=is_sparse,
    )
