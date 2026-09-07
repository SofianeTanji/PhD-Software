using Argo
using Argo.ParameterOptimization

objective = term(
    :f;
    properties=[convex(), smooth(4.0)],
    oracles=[oracle(:gradient)],
)
certificate = only(
    filter(
        candidate ->
            candidate.theorem.id === :rotaru2026_exact_fixed_step_gradient_norm,
        certificates(
            minimize(objective);
            quantity=:gradient_norm_squared,
            reformulation_depth=0,
        ),
    ),
)

free = ParameterOptimization.declared_parameters(certificate)
result = ParameterOptimization.optimize(
    [certificate],
    free;
    iterations=100,
    initial_bounds=Dict(:objective_gap => 1.0),
)

println("status: ", result.status)
println("strategy: ", result.strategy)
println("step size: ", evaluate_scalar(result.assignments[:step_size]))
println("bound: ", metric(result.assessment, :bound).value)
