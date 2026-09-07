using Argo
using Argo.Hybrid

f = term(
    :f;
    properties=[convex(), smooth(1.0e4), strongly_convex(1.0)],
    oracles=[oracle(:gradient)],
)
g = term(:g; properties=[convex()], oracles=[oracle(:prox)])
problem = minimize(f + g)
applicable = Argo.certificates(problem; reformulation_depth=0)
sapg = only(
    filter(
        certificate -> certificate.theorem.id === :taylor2017_simplified_apg,
        applicable,
    ),
)
sc_fista = only(
    filter(
        certificate ->
            certificate.theorem.id === :fista_composite_sc_gap_from_gap_derived,
        applicable,
    ),
)

plan = Hybrid.optimize(
    sapg,
    sc_fista;
    iterations=1_300,
    initial_bounds=Dict(:distance_squared => 1.0),
)
println("switch: ", plan.phases[1].iterations)
println("bound: ", metric(plan, :bound).value)
println("bound evaluations: ", metric(plan, :switch_evaluations).value)
