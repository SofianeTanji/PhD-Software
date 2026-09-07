<div align="center">
  <h1>BADR - Bilevel Adaptive Rescalarization</h1>
  <h4>Fairness-Informed Pareto Optimization</h4>
</div>

[![Python](https://img.shields.io/badge/Python-blue?logo=python&logoColor=yellow&style=for-the-badge)](https://www.python.org)
[![Scikit Learn](https://img.shields.io/badge/ScikitLearn-red?logo=scikit-learn&style=for-the-badge)](https://scikit-learn.org)
![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg?style=for-the-badge)

``badr`` is a Python package that transforms a large range of estimators into **fair** and **Pareto-efficient** estimators.

See the [documentation sources](docs/index.md) and the [experiment guide](experiments/README.md).

## Installation

This snapshot uses Python 3.11 and the dependency versions in `uv.lock`.

```sh
git clone git@github.com:SofianeTanji/PhD-Software.git
cd PhD-Software/Badr
uv sync --locked --python 3.11
uv run python -c "import badr; print(badr.__version__)"
```

Alternatively, install into a Python 3.11 virtual environment with `pip install .`.
Dataset loaders download their data when called; cached datasets are not bundled.

## Usage and experiments

See [Getting started](docs/getting-started/index.md) for the dataset, model, metric,
and `badr.Badr` interfaces. The teaching notebooks at the package root use
`lab_helpers.py`.

The experiment scripts include the solver-based BADR weight computations and the
shared `experiments/solver_utils.py` helper. Run them from their experiment
subdirectory with the environment created above. See the
[experiment guide](experiments/README.md) for solver settings and output paths.

Build the documentation locally with `uv run mkdocs build`.

## License

Released under the [BSD 3-Clause License](LICENSE).

## Citations
If you find this repository useful, or you use it in your research, please consider citing the following paper:

```
@article{tanji2026fairness,
  title   = {Fairness-informed Pareto Optimization: An Efficient Bilevel Framework},
  author  = {Tanji, Sofiane and Vaiter, Samuel and Laguel, Yassine},
  journal = {arXiv preprint arXiv:2601.13448},
  year    = {2026}
}
```
