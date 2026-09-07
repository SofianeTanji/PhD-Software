# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 2: MILP profiling (R5)
# ═══════════════════════════════════════════════════════════════════════════════
#
# For each dataset × instance × generator: build single-gen subproblem,
# count variables/constraints, time the solve.

include("common.jl")
using JuMP

function run_milp_profiling()
    BE, CA = load_instances()
    dir = joinpath(RESULTS_DIR, "milp_profiling")
    mkpath(dir)

    results = DataFrame(
        dataset = String[],
        tag = String[],
        gen = Int[],
        n_vars = Int[],
        n_constrs = Int[],
        solve_time = Float64[],
    )

    for (inst, ds, tag) in all_tagged_instances(BE, CA)
        try
            p = unpack_instance(inst)
            x0 = get_x0(inst)
            for gen = 1:p.NbGen
                model, vars = build_single_gen_subproblem(gen, p)
                # Set realistic objective using LP relaxation prices
                T = p.T
                @objective(
                    model,
                    Min,
                    sum(
                        p.NoLoadConsumption[gen] * vars.Varu[t] +
                        p.FixedCost[gen] * vars.Varv[t] +
                        (p.MarginalCost[gen] - x0[t]) * vars.Varp[t] for t = 1:T
                    )
                )
                nv = JuMP.num_variables(model)
                nc = JuMP.num_constraints(model; count_variable_in_set_constraints = false)
                t_solve = @elapsed optimize!(model)
                push!(results, (ds, tag, gen, nv, nc, t_solve))
            end
            jldsave(joinpath(dir, "milp_profiling.jld2"); results)
            @info "MILP profiling: $tag ($(p.NbGen) generators)"
        catch e
            @warn "MILP profiling $tag failed" exception=(e, catch_backtrace())
        end
    end

    @info "MILP profiling complete: $(nrow(results)) subproblems profiled."
    return results
end
