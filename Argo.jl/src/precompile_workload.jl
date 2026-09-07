@setup_workload begin
    x = variable(:x; shape=[2])
    data_fit = Atoms.least_squares([1.0 0.0; 0.0 2.0], zeros(2))(x)
    penalty = Atoms.l1(1; n=2)(x)
    problem = minimize(data_fit + penalty)

    @compile_workload begin
        found = certificates(problem; reformulation_depth=0)
        rank(
            found;
            accuracy=0.1,
            initial_bounds=Dict(:distance_squared => 1.0, :objective_gap => 1.0),
        )
    end

    # The workload must not consume user-visible declaration scopes.
    _TERM_SEQUENCE[] = 0
end
