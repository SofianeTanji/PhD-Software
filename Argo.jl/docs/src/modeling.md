# Modeling

## Native Julia objects

An objective is a closed expression algebra of `Term` values. Supported
operations are sum, difference, scalar multiplication, composition, maximum,
and minimum.

```julia
x = variable(:x; shape=[10])

f = term(
    :f;
    properties=[convex(), smooth()],
    oracles=[oracle(:value), oracle(:gradient; cost=2)],
)(x)

g = term(
    :g;
    properties=[convex(), lipschitz(1)],
    oracles=[oracle(:value), oracle(:prox)],
)(x)

problem = minimize(f + 0.1 * g)
```

Properties and oracles are distinct facts. A property describes mathematics;
an oracle declares an available operation, its exactness, and optionally its
relative cost. The built-in property calculus derives facts such as
`smooth(f + g)` from the facts on the children.

The constructor accepts `:exact`, `:controlled_inexact`, or `:inexact` and
stores the corresponding typed `OracleExactness` value. The `:any` value is
reserved for theorem requirements and cannot describe an available oracle.

Omitting a numerical constant is valid. `smooth()` creates a scoped symbolic
constant, so two unrelated terms do not accidentally share the same `L`. When
a theorem binds several roles with constants of the same name, its formula uses
role-qualified names such as `f_L` and `g_L`.

## Modeling macros

The DSL has six macros and builds the same objects as the constructors:

```julia
problem = @model begin
    @variable x[1:10]
    @term f(x)
    @term g(x)
    @assume f convex smooth()
    @oracle f value gradient
    @assume g convex
    @oracle g prox
    @minimize f(x) + g(x)
end
```

There is no maximization DSL and no parallel aliases for terms, properties, or
oracles.

## Standard atoms

Import `Argo.Atoms` when mathematical data can determine the declarations:

```julia
using Argo.Atoms

fit = least_squares(A, b)
regularizer = elastic_net(0.1, 0.01)
constraint = indicator(:simplex, length(b))
```

Atoms can inspect supplied matrices to derive constants such as operator norms
or strong convexity. They still do not store or execute numerical objectives,
gradients, proximal maps, or solvers.
