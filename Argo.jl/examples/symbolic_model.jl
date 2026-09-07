using Argo

problem = @model begin
    @variable x[1:10]
    @term f(x)
    @term g(x)
    @assume f convex smooth()
    @oracle f value gradient
    @assume g convex lipschitz(1)
    @oracle g value prox
    @minimize f(x) + g(x)
end

applicable = certificates(problem)
println("symbolic certificates: ", length(applicable))

ranked = rank(
    applicable;
    accuracy=1e-4,
    initial_bounds=Dict(:distance_squared => 1.0),
    values=Dict(:L => 2.0),
)
println("comparable after supplying L: ", length(ranked))
