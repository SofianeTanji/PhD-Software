# SNACKS

SNACKS fits binary kernel SVMs using a Nyström feature map and a restarted,
regularized accelerated stochastic subgradient solver (RASSG-r). The package
provides a scikit-learn-style estimator and scripts for the thesis experiments.

## Installation

From this directory, install the package and its dependencies:

```bash
uv sync --no-dev
# Alternatively:
# pip install -e .
```

Installation builds a Cython extension and requires a C compiler. To run the
experiment scripts, include their additional dependencies:

```bash
uv sync --no-dev --extra repro
# Alternatively:
# pip install -e ".[repro]"
```

## Quick start

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from snacks import SnacksSVM

X, y = make_classification(n_samples=500, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42,
)
clf = SnacksSVM(
    lam=1e-3,               # regularization strength
    m=50,                   # Nyström landmark count
    kernel="rbf",           # "rbf", "linear", or a callable
    gamma="median",         # median heuristic or a numeric RBF coefficient
    restarts=4,
    stages_per_restart=10,
    random_state=42,
)
clf.fit(X_train, y_train)
print("test accuracy:", clf.score(X_test, y_test))
```

Use `uv run --no-dev python` to run Python in the package environment.
Inputs are dense feature arrays with binary labels. Labels may be numbers or
strings; `predict()` returns the original labels and `decision_function()`
returns signed scores.

By default, `val_ratio=0.15` reserves part of the training data for selecting
the best stage. All configured restarts run. Setting `val_ratio=0` returns the
last stage instead. See the [algorithm guide](docs/algorithm.md) for the
objective, feature map, and restart schedule.

## Experiments

After installing the `repro` dependencies, run a small synthetic benchmark:

```bash
uv run --no-dev --extra repro python experiments/benchmark.py --dataset synthetic --smoke
```

See the [experiment guide](experiments/README.md) for datasets and experiment
commands. Data and generated results are not bundled with the package.

## Documentation

The [user guide](docs/index.md) and [algorithm guide](docs/algorithm.md) can be
read directly. To build the documentation site locally:

```bash
uv run zensical build
```

## License

Released under the [MIT license](LICENSE).
