using Argo
using Argo.Recursive

f = term(:f; properties=[convex(), smooth(1)], oracles=[oracle(:gradient)])
g = term(:g; properties=[convex(), lipschitz(1)], oracles=[oracle(:subgradient)])
problem = minimize(f + g)

direct = Argo.certificates(problem; reformulation_depth=0)
recursive = Recursive.certificates(problem; depth=1, reformulation_depth=0)

println("direct certificates: ", length(direct))
println("with recursive prox certification: ", length(recursive))
