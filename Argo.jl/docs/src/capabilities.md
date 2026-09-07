# Optional capabilities

Each first-party capability imports as a module and consumes the same core
certificates and theorem catalogue. Recursive certification returns ordinary
`Certificate` records with nested oracle evidence; comparison capabilities use
the shared `PlanAssessment` record.

## Recursive certification

```julia
using Argo.Recursive

applicable = Recursive.certificates(problem; depth=1)
```

`Recursive.Resolver` can satisfy a controlled-inexact prox requirement by
constructing its induced proximal subproblem and finding an inner certified
method. Nesting depth belongs to this module, not to the core matcher. Oracle
complexity expands the inner plan's calls into the outer plan. Numerical inner
complexity requires the corresponding `:inner_objective_gap` or
`:inner_distance_squared` value in the comparison context; Argo keeps that bound
symbolic rather than assuming a normalized starting point.

## Hybrid methods

```julia
using Argo.Hybrid

plans = Hybrid.certificates(
    problem;
    accuracy=1e-6,
    initial_bounds=Dict(:distance_squared => 1.0, :objective_gap => 1.0),
)
```

The target-accuracy mode above minimizes total oracle cost. Fixed-budget mode
minimizes the chained bound over a total number of iterations:

```julia
plans = Hybrid.certificates(
    problem;
    iterations=1_300,
    initial_bounds=Dict(:distance_squared => 1.0),
)
```

A hybrid plan is composition algebra over ordinary certificates. Pure-method
endpoints are included. Interior plans contain two `PlanPhase` records and a
`:bound`, `:switch_time`, and `:phase_one_bound`. Switch enumeration is the safe
default; binary unimodal search is used only for a built-in proven pair or an
explicit `Hybrid.UnimodalSwitchSearch` carrying provenance. Same-formulation
guarantee conversions needed by supported chains are ordinary derived theorem
declarations in the TOML catalogue.

## Sampling symbolic rankings

```julia
using Argo.Sampling

ordering = Sampling.rank(
    applicable;
    distributions=Dict(:L => Sampling.Uniform(1, 10)),
    samples=2_000,
    seed=42,
    iterations=100,
    initial_bounds=Dict(:distance_squared => 1.0),
)
```

Sampling reports the probability of being best, its standard error, expected
rank, and rank standard error through the `:probability_best` and
`:expected_rank` metrics. It never changes applicability. Distributions, sample
count, and seed are explicit.

## Parameter optimization

```julia
using Argo.ParameterOptimization

lambda = ParameterOptimization.ScalarParameter(
    :lambda, 1e-3, 10.0; scale=:log,
)
candidate = ParameterOptimization.moreau(
    problem,
    lambda;
    accuracy=1e-4,
    initial_bounds=Dict(:distance_squared => 1.0),
)
```

`explore` is explicitly a deterministic grid-and-refinement heuristic. It does
not claim a global optimum. Scientific optimization uses theorem-declared
domains and provenance-bearing registered strategies:

```julia
free = ParameterOptimization.declared_parameters(certificate)
result = ParameterOptimization.optimize(
    [certificate],
    free;
    iterations=100,
    initial_bounds=Dict(:objective_gap => 1.0),
)
result.status                 # :optimized or :not_optimized
```

Strategies may optimize several parameters jointly, but run only after their
explicit preconditions succeed. With no justified strategy, Argo returns
`:not_optimized` rather than relabeling a local numerical search as an optimum.
The Moreau convenience remains an exploratory search.

## Curvature transfer

```julia
using Argo.CurvatureTransfer

rule = CurvatureTransferRule()
applicable = certificates(
    problem;
    reformulations=Argo.Authoring.AbstractReformulationRule[rule],
    reformulation_depth=1,
)
```

The rule recognizes one explicit positive centered-quadratic summand and
replaces it by two symbolic shares in a single reformulation state. Its
enforced domain is `0 < rho < coefficient`; the unsplit formulation supplies
the endpoint groupings. This extension is exact and non-executing. It does not
infer curvature transfer from an arbitrary strongly convex black-box term.

## Regime analysis

```julia
using Argo.RegimeAnalysis

grid = regime_grid(
    applicable,
    FixedBudgetRequest(;
        iterations=100,
        initial_bounds=Dict(:distance_squared => 1.0),
    ),
    RegimeAxis(:L, [1.0, 10.0, 100.0]),
)
```

`RegimeGrid` supports one or two numerical axes, a selected metric, explicit
absolute and relative tie tolerances, and reasons for unavailable plans. Free
parameters are optimized only through justified strategies. The result is data,
not a plot; thesis-specific rendering lives in the separate reproduction
artifact.
