# Argo.jl

Argo answers one question: **which first-order minimization plans can be backed
by a convergence theorem for this problem?**

It analyzes a declarative objective, derives structural facts, binds objective
parts to theorem roles, resolves the required oracles, and returns applicable
certificates. It repeats that analysis on finite reformulations whose guarantee
can be transferred back to the initial formulation. Argo is not a numerical
solver and its atoms contain no executable implementations.

## Basic workflow

```julia
using Argo
using Argo.Atoms

x = variable(:x; shape=[2])
f = least_squares([1.0 0.0; 0.0 2.0], zeros(2))(x)
g = l1(0.1; n=2)(x)
problem = minimize(f + g)

applicable = certificates(problem)
ranked = rank(
    applicable;
    accuracy=1e-3,
    initial_bounds=Dict(:distance_squared => 1.0),
)

best = first(ranked)
phase = only(best.phases)
phase.certificate.plan.method.name
metric(best, :oracle_cost).value
```

Discovery and comparison are deliberately separate:

1. `certificates(problem)` returns only justified plans. No match, rejection,
   or diagnostic result types appear in the user workflow.
2. `rank(applicable; ...)` computes the number of iterations needed for the
   requested accuracy, expands that into role-specific oracle calls, applies
   optional oracle costs, and sorts uniform `PlanAssessment` records for the
   numerically comparable plans.

Reusable configuration can be made explicit with `ApplicabilityRequest`,
`ComparisonContext`, `RankingRequest`, and `FixedBudgetRequest`; the keyword
forms above construct the same records as a convenience. A `RankingRequest`
compares oracle cost to a target accuracy, while a `FixedBudgetRequest` compares
certified bounds after a common number of iterations.
`certificates(problem; quantity=:all)` searches every guarantee quantity
represented by the loaded catalogue.

If a constant such as smoothness is unknown, the certificate remains valid and
symbolic. It simply waits for a value before participating in numeric ranking.

## Package boundaries

- The root module contains modeling, inference, catalogue loading, certificate
  discovery, finite reformulations, and numeric oracle-complexity ranking.
- `Argo.Atoms` contains declarative standard objective atoms.
- `Argo.Recursive`, `Argo.Hybrid`, `Argo.Sampling`,
  `Argo.ParameterOptimization`, `Argo.CurvatureTransfer`, and
  `Argo.RegimeAnalysis` are opt-in capabilities built on the core protocols and
  shared theorem catalogue.
- `Argo.Authoring` contains advanced APIs for extending those protocols.

The catalogue and authoring guides define the public extension vocabulary.
