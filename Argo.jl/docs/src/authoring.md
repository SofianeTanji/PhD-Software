# Authoring extensions

Advanced extension points are grouped under `Argo.Authoring`:

```julia
using Argo.Authoring
```

- `InferenceRule` and `RuleSet` extend property or oracle propagation without
  adding cases to the expression model. Start from `default_rules()` and target
  one of the exported closed-algebra operations such as `SumOperation`.
- `AbstractOracleResolver`, `OracleQuery`, and `resolve_oracle` provide new
  `OracleEvidence` records for theorem-side oracle requirements. Custom
  evidence payloads can extend `expand_oracle_call` when they introduce nested
  work. Controlled-inexact requirements use `ControlledInexactness` and
  `InnerSolveRequirement`; extension contracts are typed rather than stored in
  metadata dictionaries.
- `AbstractReformulationRule`, `ReformulationStep`, `GuaranteeTransfer`, and
  `OutputMap` describe finite, auditable reformulations through the
  `rule_id`, `reformulate`, and `max_applications` generics.
- `load_catalogue` validates a TOML catalogue. `compose_catalogues` combines
  separately loaded catalogue modules; duplicate replacement is explicit.
- `FormulaRegistry` maps the formula names referenced by TOML to ordinary Julia
  kernels. Evaluation filters context to a kernel's explicit keyword inputs, so
  kernels do not need to accept unrelated context through `kwargs...`; a kernel
  that deliberately declares `kwargs...` receives the full compatible context.
  Custom inputs must still be named explicitly so catalogue validation can
  trace their provenance. The same registry can hold named guarantee-transfer
  formulas, which receive a symbolic `value` and the transfer's declared
  parameters.
- `OracleExactness` centralizes the closed exact, controlled-inexact, inexact,
  and requirement-only any contracts. `OracleRolePath` retains composed-work
  provenance without encoding it into a delimiter-separated role name.
- Certificate accessors such as `method_declaration`, `role_binding`,
  `oracle_evidence`, `source_formulation`, `reformulation_path`, and
  `uses_scalar` let
  extensions consume the stable certificate protocol without depending on its
  storage layout.

These APIs are intentionally not exported from the root namespace. Most users
only need `Term`, `Property`, `Oracle`, `certificates`, and `rank`.
