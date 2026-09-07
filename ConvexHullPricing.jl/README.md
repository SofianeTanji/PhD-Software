# ConvexHullPricing


## Overview
The `ConvexHullPricing.jl` package implements various primal and dual optimization methods to compute Convex Hull Prices, relying on a structure model to load thermal generators and unit commitment instances. It is primarily designed to benchmark optimization methods for this problem but one can also compute various metrics that are useful to practitioners.

## Installation

Requires Julia 1.10 or later and a working Gurobi installation/license. The package
creates a Gurobi environment when imported.

Clone the software collection and open the package environment:

```sh
git clone git@github.com:SofianeTanji/PhD-Software.git
cd PhD-Software/ConvexHullPricing.jl
julia --project=.
```

```julia
using Pkg
Pkg.instantiate()
using ConvexHullPricing
```

## Computing prices and uplift

Run from the `ConvexHullPricing.jl` directory:

```julia
instance = load_data("data/belgian/belgian-autumnwd.json")
prices = compute_ch_prices(instance; budget=60.0)
market_schedule = solve_market_schedule(instance)
uplift = total_uplift(instance, prices, market_schedule)
report = settlement_report(instance, prices, market_schedule)
```

The default pricing method is column generation. Select another method with the
`method` keyword, for example `method=:preconditioned_proximal` for PC-BPLM.
A floating-point budget denotes seconds; an integer or `IterationLimit(n)` denotes
an iteration limit. Scheduling and settlement computations run separately from
the pricing budget. Californian instances use `load_ca_data(path)`.

## Experiments

Input instances are included under `data/`. To use the experiment scripts, start
Julia from the package directory with `julia --project=experiments`, then run:

```julia
using Pkg
Pkg.develop(path=".")
Pkg.instantiate()
include("experiments/full_experiments.jl")
run_main_benchmark()
```

The scripts write generated outputs under `results/`. Individual experiments can
be called after including the driver; `run_all()` executes the reference,
benchmark and figure pipeline. Reference computations can take hours per instance.

## Citing this package

If you use `ConvexHullPricing.jl` for published work, we encourage you to cite the software using the following Bibtex citation:
```bibtex
@misc{tanji2025dualfirstordermethodsefficient,
      title={Dual first-order methods for efficient computation of convex hull prices}, 
      author={Sofiane Tanji and Yassine Kamri and François Glineur and Mehdi Madani},
      year={2025},
      eprint={2504.01474},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2504.01474}, 
}
```
## License
`ConvexHullPricing.jl` is released under a MIT License. `ConvexHullPricing.jl` has been developed as part of the ITN-ETN project TraDE-OPT funded by the European Union’s Horizon 2020 research and innovation programme under the Marie Sklodowska-Curie grant agreement No 861137.

## Get in touch
Comments and suggestions are more than welcome, get in touch via [mail](mailto:sofiane.tanji@uclouvain.be) or an issue!
