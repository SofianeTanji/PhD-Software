# Catalogue and certificates

## One source of theorem truth

Method declarations live in `catalogue/methods.toml`; theorem declarations live
in `catalogue/theorems/*.toml`. TOML is the only tracked source of theorem data.
Argo validates unknown fields, duplicate identifiers, role/call consistency,
known quantities, named formula references, and the declared keyword inputs of
custom formula registrations while loading it. A formula input must come from
an initial bound, a selected method parameter, or a required property or oracle
contract. Validation errors carry the source file and entry number.

A method declaration owns the method name, parameter vocabulary, required
recipe entries, and per-iteration oracle-call profile. A theorem declaration
separately owns role requirements, its guarantee quantity and output
convention, its initial bound, its bound formula, and its citation. Several
theorems may therefore justify the same method without duplicating its
operational profile.

`parameter_vocabulary` is the closed set of parameter names that the method may
use, while `required_parameters` names the entries every theorem for the method
must supply. A theorem selects either a fixed formula in `parameter_rules` or a
free admissible interval in `parameter_domains`; the same name cannot occur in
both. Domain endpoints are finite literals or registered formula names and may
be open or closed. Loading rejects a theorem that omits a required entry,
selects an undeclared name, leaves a method parameter unused, or references a
named formula without a registered Julia implementation. Consequently, the
`MethodPlan` contains the complete parameter contract, and free values remain
symbolic until comparison.
Dependencies among selected parameters are resolved independent of declaration
order, and circular recipes are rejected while loading.
Accelerated methods select both their step and inertial coefficient; scheduled
values are queried with `parameter_value(certificate, :momentum; k=...)`.
Free ranges are queried with `parameter_interval(certificate, :step_size)` and
are enforced whenever a parameter or bound is evaluated. For example, the
Rotaru free-step gradient theorem declares the strict interval
`0 < step_size < 2/L` in TOML rather than hiding it in experiment code.

## Applicability

```julia
applicable = certificates(
    problem;
    quantity=:objective_gap,
    reformulation_depth=1,
)

all_quantities = certificates(problem; quantity=:all)
```

For each theorem, Argo generates labeled role bindings from the objective,
checks the required properties, and asks an oracle resolver for evidence. A
certificate is returned only when every condition succeeds. Different valid
role bindings remain different certificates because their constants and oracle
costs may differ.

Property constants are always retained under a fully qualified name such as
`f__smooth__L`. Argo also supplies `f_L` when that name is unique within the
role, and the familiar `L` when it is unique across the theorem. A theorem with
two smooth roles therefore uses `f_L` and `g_L`; it never silently equates their
constants. An open custom property uses the fully qualified form—for example,
`f__custom_curvature__C`—to state the input's provenance in its formula.
Supplying an unqualified numerical value only substitutes an unambiguous
symbol; use `f_L` and `g_L` separately when both occur.

The default reformulation search includes the initial formulation and one level
of valid reformulations. Every edge has a typed guarantee transfer and output
map; a transformed guarantee is never silently treated as a guarantee on the
initial formulation.

## Ranking by oracle complexity

```julia
ranked = rank(
    applicable;
    accuracy=1e-5,
    initial_bounds=Dict(:distance_squared => 4.0),
    oracle_costs=Dict(
        (:f, :gradient) => 2.0,
        (:g, :prox) => 10.0,
    ),
    values=Dict(:L => 3.0),
)
```

For each comparable certificate, Argo inverts its non-asymptotic bound to find
the iteration count, multiplies the method's per-iteration calls by that count,
then applies role-specific or global oracle costs. `OracleComplexity` preserves
the call profile. Work introduced by a reformulation is expressed using the
source problem's oracles, including the final output-map operation. Ranking
returns `PlanAssessment` records; the scalar used for
ordering is `metric(assessment, :oracle_cost).value`, and the constituent
certificate and iteration count are in `assessment.phases`. Certificates that
remain symbolic are omitted rather than assigned an arbitrary order.

For primal-dual methods on `g(Ax)`, the profile names the applications of `A`
and `A'` explicitly as `operator` and `adjoint`, in addition to the objective
oracles. A genuine saddle-point theorem reports `primal_dual_gap` from a
`primal_dual_distance_squared` initial bound; it is not presented as a primal
objective-gap theorem.

Nested and hybrid work has a structured role path. Its printed label remains
compact (for example, `phase1__f`), while role-specific cost fallback uses the
recorded source role rather than parsing that label.

For a common iteration budget, use the distinct fixed-budget request:

```julia
fixed = rank(
    applicable,
    FixedBudgetRequest(;
        iterations=100,
        initial_bounds=Dict(:distance_squared => 4.0),
    ),
)
metric(first(fixed), :bound).value
```

This mode orders the numerical worst-case bounds and retains fixed-budget
oracle work as a secondary metric. It does not invert bounds to an accuracy.
