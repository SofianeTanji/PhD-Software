using Argo
using Argo.Atoms

A = [1.0 0.0; 0.0 2.0]
b = zeros(2)
x = variable(:x; shape=[2])

problem = minimize(least_squares(A, b)(x) + l1(0.1; n=2)(x))

applicable = certificates(problem)
ranked = rank(
    applicable;
    accuracy=1e-3,
    initial_bounds=Dict(:distance_squared => 1.0),
    oracle_costs=Dict(:gradient => 1.0, :prox => 2.0),
)

println("applicable certificates: ", length(applicable))
for candidate in first(ranked, min(5, length(ranked)))
    println(candidate)
    println("  ", candidate.complexity)
end
