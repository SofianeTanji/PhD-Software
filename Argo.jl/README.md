# Argo.jl

Argo discovers first-order minimization methods that come with a theorem-backed
guarantee for a structured objective. It checks the formulation you wrote and
finite, guarantee-preserving reformulations. It does not run the methods or
solve the optimization problem.

## Quick start

Describe the objective with native Julia constructors:

```julia
using Argo
using Argo.Atoms

A = [1.0 0.0; 0.0 2.0]
b = zeros(2)
x = variable(:x; shape=[2])

fit = least_squares(A, b)(x)
penalty = l1(0.1; n=2)(x)
problem = minimize(fit + penalty)

applicable = certificates(problem)
ranked = rank(
    applicable;
    accuracy=1e-3,
    initial_bounds=Dict(:distance_squared => 1.0),
    oracle_costs=Dict(:gradient => 1.0, :prox => 2.0),
)

best = first(ranked)
phase = only(best.phases)
phase.certificate.plan.method.name
phase.iterations
best.complexity
metric(best, :oracle_cost).value
```

`certificates` contains only plans that Argo can justify. An empty vector means
that the loaded catalogue contains no applicable theorem. Pass `quantity=:all`
to discover every guarantee quantity represented by the catalogue. `rank` is a
separate step and returns uniform `PlanAssessment` records; it omits certificates
whose symbolic constants have not yet been given enough numerical information
to compare them.

## Familiar modeling syntax

The six modeling macros are a thin front end over the same Julia objects:

```julia
problem = @model begin
    @variable x[1:50]
    @term f(x)
    @term g(x)

    @assume f convex smooth()       # an unknown, scoped smoothness constant
    @oracle f value gradient

    @assume g convex lipschitz(1.0)
    @oracle g value prox

    @minimize f(x) + g(x)
end

applicable = certificates(problem)
ranked = rank(
    applicable;
    accuracy=1e-4,
    initial_bounds=Dict(:distance_squared => 1.0),
    values=Dict(:L => 2.0),
)
```

The canonical oracle names are `value`, `gradient`, `subgradient`, `prox`, and
`linear_minimization`. Atoms and oracles are declarations only: Argo never asks
for executable objective or oracle functions.

## Optional capabilities

The small core is extended through explicit submodules:

```julia
using Argo.Recursive               # certify an inexact oracle with an inner plan
using Argo.Hybrid                  # compose compatible two-phase certificates
using Argo.Sampling                # Monte Carlo ordering of symbolic bounds
using Argo.ParameterOptimization   # justified joint parameter optimization
using Argo.CurvatureTransfer       # redistribute explicit centered curvature
using Argo.RegimeAnalysis          # compute renderer-independent regime grids
```

Advanced catalogue, inference-rule, reformulation, and oracle-resolver APIs live
under `Argo.Authoring`. The theorem declarations themselves have one tracked
source of truth: the TOML files in [`catalogue/`](catalogue/).

See [`docs/src/index.md`](docs/src/index.md) for the package guide and
[`docs/src/catalogue.md`](docs/src/catalogue.md) for the catalogue vocabulary.
