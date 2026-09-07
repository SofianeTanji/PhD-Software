module ConvexHullPricing

using DataFrames
using Gurobi
using JSON3
using JuMP
using LinearAlgebra

const GRB_ENV = Ref{Gurobi.Env}()

function __init__()
    GRB_ENV[] = Gurobi.Env()
    return
end

const PCU = 300.0
const PCD = -300.0
const PC = abs(PCU) + abs(PCD)

ProjBox(v, lb, ub) = clamp.(v, lb, ub)

include("utilities/dataloader.jl")
include("utilities/structures.jl")
include("utilities/model_helpers.jl")
include("utilities/oracles.jl")

include("optimizers/stopping.jl")
include("optimizers/preconditioners.jl")

include("optimizers/dual_methods/BundleLevelMethod.jl")
include("optimizers/dual_methods/BundleProximalMethod.jl")
include("optimizers/dual_methods/ZhangBundleMethods.jl")
include("optimizers/dual_methods/DAdaptation.jl")
include("optimizers/dual_methods/DoWG.jl")
include("optimizers/dual_methods/FastGradientMethod.jl")
include("optimizers/dual_methods/PolyakMethod.jl")
include("optimizers/dual_methods/SubgradientMethod.jl")
include("optimizers/dual_methods/StochasticMethods.jl")

include("optimizers/primal_methods/ColumnGeneration.jl")
include("utilities/settlement.jl")

export load_data, load_ca_data

export ThermalGen, Instance
export SubProblem, RestrictedMasterProgram

export PCU, PCD, PC, ProjBox

export unpack_instance, new_gurobi_model
export build_single_gen_subproblem, build_joint_uc_model, build_cg_subproblem
export joint_production_cost, lagrangian_gen_cost
export extract_gen_gradient
export origin_prox_term
export demand_block_closedform, build_gen_subproblem
export gen_smoothed_objective, gen_ponly_smoothed_objective, extract_gen_gradient_no_load

export LP_Relaxation, GetShift, fast_uc_upper_bound
export ExactOracleCache, build_oracle_subproblems
export exact_oracle
export exact_smooth_oracle, exact_translate_smooth_oracle
export exact_oracle_multicut, exact_translate_smooth_oracle_multicut
export compute_ch_prices, solve_market_schedule, self_schedule
export unit_uplifts, total_uplift, settlement_report
export StoppingCriterion, IterationLimit, TimeBudget, TimeBudgetWithLstar, GapTolerance, GapToleranceWithBudget
export should_continue, max_iterations, negate_for_minimization

export BundleLevelMethod, MulticutBundleLevelMethod
export BundleProximalLevelMethod, MulticutBundleProximalLevelMethod
export PreconditionedLevelMethod, PreconditionedProximalLevelMethod #  PC-BLM ; PC-BPLM ; DLM ; DA ; DOWG ; FGM ; POLYAK ; SUBG ; EST-POLYAK ; L-SUBG ; CG
export PreconditionerConfig, ClippedInversePrecond
export DynamicLevelMethod
export DAdaptation, DowG
export FastGradientMethod
export PolyakMethod
export SubgradientMethod
export EstimatedPolyak, LastIterateSubgradientMethod

export ColumnGeneration

end # module ConvexHullPricing
