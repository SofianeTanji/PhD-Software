# Snacks

Kernel SVM classification using a Nyström feature map and RASSG-r optimization.

## Install

From the Snacks directory:

```bash
pip install -e .
```

A C compiler is required to build the Cython extension. Use
`pip install -e ".[repro]"` to include the experiment dependencies.

## Quick start

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from snacks import SnacksSVM

X, y = make_classification(n_samples=500, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42,
)
clf = SnacksSVM(lam=1e-3, m=50, restarts=4, stages_per_restart=10, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("test accuracy:", clf.score(X_test, y_test))
```

The estimator accepts dense feature arrays and any two distinct label values.
It supports cloning and classifier detection through scikit-learn.

## How it works

1. A Nyström embedding approximates the kernel using landmark points sampled
   uniformly from the training data.
2. RASSG-r optimizes the regularized hinge loss through a sequence of centered
   stochastic subgradient stages and outer restarts.
3. With the default validation split, the estimator returns the stage with the
   highest validation accuracy after completing the restart schedule. With
   `val_ratio=0`, it returns the final stage.

See the [algorithm guide](algorithm.md) for the objective and default schedule.
