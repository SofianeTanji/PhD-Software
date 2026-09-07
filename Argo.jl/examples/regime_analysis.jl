using Argo
using Argo.RegimeAnalysis

objective = term(
    :f;
    properties=[convex(), smooth()],
    oracles=[oracle(:gradient)],
)
applicable = certificates(minimize(objective); reformulation_depth=0)
request = FixedBudgetRequest(;
    iterations=100,
    initial_bounds=Dict(:distance_squared => 1.0),
)
grid = regime_grid(
    applicable,
    request,
    RegimeAxis(:L, [1.0, 10.0, 100.0]);
    optimize_parameters=false,
)

for cell in grid.cells
    winners = [
        only(assessment.phases).certificate.theorem.id for
        assessment in cell.winners
    ]
    println(cell.coordinates, " => ", winners)
end
