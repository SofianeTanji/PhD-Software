# PhD software

Software accompanying my PhD thesis.

- [Argo.jl](Argo.jl/): a Julia package for discovering and comparing first-order optimization methods through convergence certificates.
- [Snacks](Snacks/): a Python package for binary kernel SVMs using Nyström features and RASSG-r optimization.
- [ConvexHullPricing.jl](ConvexHullPricing.jl/): a Julia toolbox for convex hull pricing, unit commitment schedules, and uplift calculations.
- [Badr](Badr/): a Python toolbox for fairness-informed Pareto optimization using bilevel adaptive rescalarization.

Each software directory contains its documentation and license.

## Using Argo.jl

Clone this repository, then open Julia with the package environment:

```sh
git clone git@github.com:SofianeTanji/PhD-Software.git
cd PhD-Software/Argo.jl
julia --project=.
```

In Julia:

```julia
using Pkg
Pkg.instantiate()
using Argo
```

See the [Argo.jl guide](Argo.jl/README.md) for examples.

## Using Snacks

From the cloned repository:

```sh
cd Snacks
uv sync --no-dev --extra repro
```

See the [Snacks guide](Snacks/README.md) for installation alternatives and examples.

## Using ConvexHullPricing.jl

From the cloned repository:

```sh
cd ConvexHullPricing.jl
julia --project=.
```

Run `using Pkg; Pkg.instantiate()` in Julia. A working Gurobi license is required.
See the [ConvexHullPricing.jl guide](ConvexHullPricing.jl/README.md) for usage and experiments.

## Using Badr

From the cloned repository:

```sh
cd Badr
uv sync --locked --python 3.11
```

See the [Badr guide](Badr/README.md) for usage, experiments, and documentation.
