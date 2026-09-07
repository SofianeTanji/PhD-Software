using Argo
using Argo.Atoms
using Argo.Hybrid

problem = minimize(sum_squares())
plans = Hybrid.certificates(
    problem;
    accuracy=1e-3,
    initial_bounds=Dict(:distance_squared => 1.0, :objective_gap => 1.0),
    max_switch=100,
)

println("hybrid plans: ", length(plans))
isempty(plans) || println("best: ", first(plans))
