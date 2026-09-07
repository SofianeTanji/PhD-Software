# PhD software

Software accompanying my PhD thesis.

- [Argo.jl](Argo.jl/): a Julia package for discovering and comparing first-order optimization methods through convergence certificates.
- [Snacks](Snacks/): a Python package for binary kernel SVMs using Nyström features and RASSG-r optimization.

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
