"""
    unpack_instance(instance) -> NamedTuple

Extract all fields from an Instance into a flat NamedTuple for convenient access.
Replaces the 14-line "unzip" block that was copy-pasted in every oracle function.
"""
function unpack_instance(instance)
    tg = instance.ThermalGen
    L = instance.Load
    return (
        MinRunCapacity = tg.MinRunCapacity,
        MaxRunCapacity = tg.MaxRunCapacity,
        RampUp = tg.RampUp,
        RampDown = tg.RampDown,
        UpTime = tg.UpTime,
        DownTime = tg.DownTime,
        StartUp = tg.StartUp,
        ShutDown = tg.ShutDown,
        NbGen = length(tg.MinRunCapacity),
        FixedCost = tg.FixedCost,
        MarginalCost = tg.MarginalCost,
        NoLoadConsumption = tg.NoLoadConsumption,
        L = L,
        T = length(L),
        LostLoad = instance.LostLoad,
    )
end

"""
    new_gurobi_model(; mip_gap=0, mip_gap_abs=1e-8) -> Model

Create a new silent Gurobi direct model with standard MIP tolerances.
"""
function new_gurobi_model(; mip_gap = 0, mip_gap_abs = 1e-8)
    model = JuMP.direct_model(Gurobi.Optimizer(GRB_ENV[]))
    set_silent(model)
    set_optimizer_attributes(model, "MIPGap" => mip_gap, "MIPGapAbs" => mip_gap_abs)
    return model
end

"""
    add_single_gen_vars!(model, T, L) -> NamedTuple

Add the standard single-generator UC variables and load variable to a JuMP model.
Returns `(; Varp, Varu, Varv, Varw, VarL)`.
"""
function add_single_gen_vars!(model, T, L)
    Varp = @variable(model, [t = 0:(T+1)], lower_bound = 0)
    Varu = @variable(model, [t = 0:(T+1)], Bin)
    Varv = @variable(model, [t = 0:(T+1)], Bin)
    Varw = @variable(model, [t = 0:(T+1)], Bin)
    VarL = @variable(model, [t = 1:T], lower_bound = 0)
    @constraint(model, [t = 1:T], VarL[t] <= L[t])
    return (; Varp, Varu, Varv, Varw, VarL)
end

"""
    add_single_gen_constraints!(model, vars, gen, p) -> Nothing

Add the standard UC constraint groups for a single generator.
`vars` is the output of `add_single_gen_vars!`, `gen` is the generator index,
and `p` is the unpacked instance NamedTuple from `unpack_instance`.
"""
function add_single_gen_constraints!(model, vars, gen, p)
    T = p.T
    (; Varp, Varu, Varv, Varw) = vars

    @constraint(model, [t = 1:(T+1)], Varu[t] - Varu[t-1] == Varv[t] - Varw[t])

    @constraint(
        model,
        [t = p.UpTime[gen]:(T+1)],
        sum(Varv[i] for i = (t-p.UpTime[gen]+1):t) <= Varu[t]
    )

    @constraint(
        model,
        [t = p.DownTime[gen]:(T+1)],
        sum(Varw[i] for i = (t-p.DownTime[gen]+1):t) <= (1 - Varu[t])
    )

    @constraint(model, [t = 0:(T+1)], p.MinRunCapacity[gen] * Varu[t] <= Varp[t])

    @constraint(model, [t = 0:(T+1)], Varp[t] <= p.MaxRunCapacity[gen] * Varu[t])

    @constraint(
        model,
        [t = 1:(T+1)],
        Varp[t] - Varp[t-1] <= p.RampUp[gen] * Varu[t-1] + p.StartUp[gen] * Varv[t]
    )

    @constraint(
        model,
        [t = 1:(T+1)],
        Varp[t-1] - Varp[t] <= p.RampDown[gen] * Varu[t] + p.ShutDown[gen] * Varw[t]
    )

    return nothing
end

"""
    build_single_gen_subproblem(gen, p; mip_gap=0, mip_gap_abs=1e-8) -> (model, vars)

Create a complete single-generator UC subproblem (model + variables + constraints).
Convenience wrapper combining `new_gurobi_model`, `add_single_gen_vars!`,
and `add_single_gen_constraints!`.
"""
function build_single_gen_subproblem(gen, p; mip_gap = 0, mip_gap_abs = 1e-8)
    model = new_gurobi_model(; mip_gap, mip_gap_abs)
    vars = add_single_gen_vars!(model, p.T, p.L)
    add_single_gen_constraints!(model, vars, gen, p)
    return model, vars
end

"""
    lagrangian_gen_cost(p, gen, vars, prices) -> JuMP expression

The standard per-generator Lagrangian objective (without any smoothing term).
Used by `exact_oracle`, and `soracle`.
"""
function lagrangian_gen_cost(p, gen, vars, prices)
    (; Varp, Varu, Varv, VarL) = vars
    T, NbGen = p.T, p.NbGen
    return sum(
        p.NoLoadConsumption[gen] * Varu[t] +
        p.FixedCost[gen] * Varv[t] +
        p.MarginalCost[gen] * Varp[t] +
        (p.LostLoad / NbGen) * (p.L[t] - VarL[t]) -
        prices[t] * (Varp[t] - (VarL[t] / NbGen)) for t = 1:T
    )
end

"""
    extract_gen_gradient(vars, T, NbGen) -> Vector{Float64}

Compute the (sub)gradient contribution `VarL[t]/NbGen - Varp[t]` for one generator.
This is the gradient of the Lagrangian dual w.r.t. prices for a single generator.

Note: the original smooth oracles computed this via a matrix product `Aᵀ U` with
A = [I; 0; 0; 0; 0; (-1/NbGen)I], which simplifies to the same expression.
"""
function extract_gen_gradient!(out::AbstractVector{Float64}, vars, T, NbGen)
    @inbounds for t = 1:T
        out[t] = value(vars.VarL[t]) / NbGen - value(vars.Varp[t])
    end
    return out
end

function extract_gen_gradient(vars, T, NbGen)
    grad = Vector{Float64}(undef, T)
    extract_gen_gradient!(grad, vars, T, NbGen)
    return grad
end

"""
    origin_prox_term(vars, T) -> JuMP expression

Origin-centered quadratic proximal term: Σ (p² + u² + v² + w² + L²).
Used by `exact_smooth_oracle` and.
"""
function origin_prox_term(vars, T)
    (; Varp, Varu, Varv, Varw, VarL) = vars
    return sum(Varp[t]^2 + Varu[t]^2 + Varv[t]^2 + Varw[t]^2 + VarL[t]^2 for t = 1:T)
end


"""
    build_joint_uc_model(p; binary=true, balance=true, mip_gap=1e-3)
        -> (model, vars)

Build a full multi-generator UC model with all constraints.

- `binary=true`: use binary variables for u, v, w (MIP); `false` = LP relaxation [0,1].
- `balance=true`: include the load balance constraint `Σ p[g,t] == L[t]`.
- Returns `(model, vars)` where `vars` is a NamedTuple with `Varp, Varu, Varv, Varw, VarL, loads`.
  `loads` is the balance constraint reference (or `nothing` if `balance=false`).
"""
function build_joint_uc_model(p; binary = true, balance = true, mip_gap = 1e-3)
    T, NbGen = p.T, p.NbGen
    L = p.L

    model = new_gurobi_model(; mip_gap)

    Varp = @variable(model, [g = 1:NbGen, t = 0:(T+1)], lower_bound = 0)
    if binary
        Varu = @variable(model, [g = 1:NbGen, t = 0:(T+1)], Bin)
        Varv = @variable(model, [g = 1:NbGen, t = 0:(T+1)], Bin)
        Varw = @variable(model, [g = 1:NbGen, t = 0:(T+1)], Bin)
    else
        Varu =
            @variable(model, [g = 1:NbGen, t = 0:(T+1)], lower_bound = 0, upper_bound = 1)
        Varv =
            @variable(model, [g = 1:NbGen, t = 0:(T+1)], lower_bound = 0, upper_bound = 1)
        Varw =
            @variable(model, [g = 1:NbGen, t = 0:(T+1)], lower_bound = 0, upper_bound = 1)
    end
    VarL = @variable(model, [t = 1:T], lower_bound = 0)
    @constraint(model, [t = 1:T], VarL[t] <= L[t])

    loads = nothing
    if balance
        loads = @constraint(model, [t = 1:T], sum(Varp[g, t] for g = 1:NbGen) == VarL[t])
    end

    @constraint(
        model,
        [gen = 1:NbGen, t = 1:(T+1)],
        Varu[gen, t] - Varu[gen, t-1] == Varv[gen, t] - Varw[gen, t]
    )
    @constraint(
        model,
        [gen = 1:NbGen, t = p.UpTime[gen]:(T+1)],
        sum(Varv[gen, i] for i = (t-p.UpTime[gen]+1):t) <= Varu[gen, t]
    )
    @constraint(
        model,
        [gen = 1:NbGen, t = p.DownTime[gen]:(T+1)],
        sum(Varw[gen, i] for i = (t-p.DownTime[gen]+1):t) <= (1 - Varu[gen, t])
    )
    @constraint(
        model,
        [gen = 1:NbGen, t = 0:(T+1)],
        p.MinRunCapacity[gen] * Varu[gen, t] <= Varp[gen, t]
    )
    @constraint(
        model,
        [gen = 1:NbGen, t = 0:(T+1)],
        Varp[gen, t] <= p.MaxRunCapacity[gen] * Varu[gen, t]
    )
    @constraint(
        model,
        [gen = 1:NbGen, t = 1:(T+1)],
        Varp[gen, t] - Varp[gen, t-1] <=
        p.RampUp[gen] * Varu[gen, t-1] + p.StartUp[gen] * Varv[gen, t]
    )
    @constraint(
        model,
        [gen = 1:NbGen, t = 1:(T+1)],
        Varp[gen, t-1] - Varp[gen, t] <=
        p.RampDown[gen] * Varu[gen, t] + p.ShutDown[gen] * Varw[gen, t]
    )

    return model, (; Varp, Varu, Varv, Varw, VarL, loads)
end

"""
    joint_production_cost(p, vars) -> JuMP expression

Standard production cost objective for the joint multi-generator model:
  Σ_gen Σ_t (NoLoad·u + Fixed·v + Marginal·p) + LostLoad · Σ_t (L - VarL)
"""
function joint_production_cost(p, vars)
    (; Varp, Varu, Varv, VarL) = vars
    return sum(
        sum(
            p.NoLoadConsumption[gen] * Varu[gen, t] +
            p.FixedCost[gen] * Varv[gen, t] +
            p.MarginalCost[gen] * Varp[gen, t] for t = 1:p.T
        ) for gen = 1:p.NbGen
    ) + p.LostLoad * sum(p.L[t] - VarL[t] for t = 1:p.T)
end


"""
    build_cg_subproblem(gen, p) -> SubProblem

Build a single-generator subproblem for column generation. This creates the
same UC variables and constraints as `build_single_gen_subproblem`, but without
`VarL` (no lost load in CG subproblems) and with an additional `VarCost`
variable and named constraint references needed by the `SubProblem` struct.
"""
function build_cg_subproblem(gen, p)
    T = p.T
    model = new_gurobi_model()

    Varp = @variable(model, [t = 0:(T+1)], lower_bound = 0)
    Varu = @variable(model, [t = 0:(T+1)], Bin)
    Varv = @variable(model, [t = 0:(T+1)], Bin)
    Varw = @variable(model, [t = 0:(T+1)], Bin)
    VarCost = @variable(model)

    ConstrLogical =
        @constraint(model, [t = 1:(T+1)], Varu[t] - Varu[t-1] == Varv[t] - Varw[t])
    ConstrMinUpTime = @constraint(
        model,
        [t = p.UpTime[gen]:(T+1)],
        sum(Varv[i] for i = (t-p.UpTime[gen]+1):t) <= Varu[t]
    )
    ConstrMinDownTime = @constraint(
        model,
        [t = p.DownTime[gen]:(T+1)],
        sum(Varw[i] for i = (t-p.DownTime[gen]+1):t) <= (1 - Varu[t])
    )
    ConstrGenLimits1 =
        @constraint(model, [t = 0:(T+1)], p.MinRunCapacity[gen] * Varu[t] <= Varp[t])
    ConstrGenLimits2 =
        @constraint(model, [t = 0:(T+1)], Varp[t] <= p.MaxRunCapacity[gen] * Varu[t])
    ConstrRampUp = @constraint(
        model,
        [t = 1:(T+1)],
        Varp[t] - Varp[t-1] <= p.RampUp[gen] * Varu[t-1] + p.StartUp[gen] * Varv[t]
    )
    ConstrRampDown = @constraint(
        model,
        [t = 1:(T+1)],
        Varp[t-1] - Varp[t] <= p.RampDown[gen] * Varu[t] + p.ShutDown[gen] * Varw[t]
    )

    @constraint(
        model,
        VarCost - sum(
            p.NoLoadConsumption[gen] * Varu[t] +
            p.FixedCost[gen] * Varv[t] +
            p.MarginalCost[gen] * Varp[t] for t = 1:T
        ) == 0
    )

    return SubProblem(
        model,
        Varp,
        Varu,
        Varv,
        Varw,
        VarCost,
        ConstrLogical,
        ConstrMinUpTime,
        ConstrMinDownTime,
        ConstrGenLimits1,
        ConstrGenLimits2,
        ConstrRampUp,
        ConstrRampDown,
    )
end


"""
    demand_block_closedform(p, prices, ς, shiftL) -> (obj, grad, l_star)

Closed-form demand block L_{0,ς}(π) from eq. (15).
Computes the proximal projection of lost-load variable analytically per period.
"""
function demand_block_closedform(p, prices, ς, shiftL)
    obj = 0.0
    grad = zeros(p.T)
    l_star = zeros(p.T)
    for t = 1:p.T
        l_star[t] = clamp(shiftL[t] - (prices[t] - p.LostLoad) / ς, 0.0, p.L[t])
        obj +=
            (prices[t] - p.LostLoad) * l_star[t] +
            p.LostLoad * p.L[t] +
            (ς / 2) * (l_star[t] - shiftL[t])^2
        grad[t] = l_star[t]
    end
    return obj, grad, l_star
end

"""
    add_gen_vars!(model, T) -> NamedTuple

Add single-generator UC variables (p, u, v, w) without VarL.
"""
function add_gen_vars!(model, T)
    Varp = @variable(model, [t = 0:(T+1)], lower_bound = 0)
    Varu = @variable(model, [t = 0:(T+1)], Bin)
    Varv = @variable(model, [t = 0:(T+1)], Bin)
    Varw = @variable(model, [t = 0:(T+1)], Bin)
    return (; Varp, Varu, Varv, Varw)
end

"""
    build_gen_subproblem(gen, p; mip_gap=0, mip_gap_abs=1e-8) -> (model, vars)

Build a single-generator UC subproblem without VarL.
Uses `add_gen_vars!` + `add_single_gen_constraints!`.
"""
function build_gen_subproblem(gen, p; mip_gap = 0, mip_gap_abs = 1e-8)
    model = new_gurobi_model(; mip_gap, mip_gap_abs)
    vars = add_gen_vars!(model, p.T)
    add_single_gen_constraints!(model, vars, gen, p)
    return model, vars
end

"""
    gen_smoothed_objective(p, gen, vars, prices, ς, shift) -> JuMP expression

Per-generator smoothed Lagrangian objective L_{g,ς}(π) from eq. (16).
Proximal on binaries is linearized via x²=x.
"""
function gen_smoothed_objective(p, gen, vars, prices, ς, shift)
    (; Varp, Varu, Varv, Varw) = vars
    T = p.T
    shiftU, shiftV, shiftW, shiftP, shiftL = shift
    NbGen = p.NbGen

    return sum(
        # Quadratic proximal on continuous p
        (ς / 2) * (Varp[t] - shiftP[gen, t])^2 +
        # Lagrangian cost: (MarginalCost - π_t) * p_t
        (p.MarginalCost[gen] - prices[t]) * Varp[t] +
        # Linearized proximal on u: (NoLoad + (ς/2)(1 - 2·u^r)) · u
        (p.NoLoadConsumption[gen] + (ς / 2) * (1 - 2 * shiftU[gen, t])) * Varu[t] +
        # Linearized proximal on v: (Fixed + (ς/2)(1 - 2·v^r)) · v
        (p.FixedCost[gen] + (ς / 2) * (1 - 2 * shiftV[gen, t])) * Varv[t] +
        # Linearized proximal on w: (ς/2)(1 - 2·w^r) · w
        (ς / 2) * (1 - 2 * shiftW[gen, t]) * Varw[t] +
        # Constants from linearization: (ς/2)(u^r² + v^r² + w^r²)
        (ς / 2) * (shiftU[gen, t]^2 + shiftV[gen, t]^2 + shiftW[gen, t]^2) for t = 1:T
    )
end

"""
    gen_ponly_smoothed_objective(p, gen, vars, prices, ς, shift) -> JuMP expression

Per-generator smoothed Lagrangian objective with proximal term on dispatch (p) only.
No smoothing on binary variables u, v, w — they keep their standard Lagrangian cost.
"""
function gen_ponly_smoothed_objective(p, gen, vars, prices, ς, shift)
    (; Varp, Varu, Varv, Varw) = vars
    T = p.T
    _, _, _, shiftP, _ = shift
    return sum(
        # Quadratic proximal on continuous p
        (ς / 2) * (Varp[t] - shiftP[gen, t])^2 +
        # Lagrangian cost: (MarginalCost - π_t) * p_t
        (p.MarginalCost[gen] - prices[t]) * Varp[t] +
        # Standard binary costs (no proximal)
        p.NoLoadConsumption[gen] * Varu[t] +
        p.FixedCost[gen] * Varv[t] for t = 1:T
    )
end

"""
    extract_gen_gradient_no_load(vars, T) -> Vector{Float64}

Gradient contribution from a single generator (no VarL term): -p_g[t].
"""
function extract_gen_gradient_no_load(vars, T)
    return [-value(vars.Varp[t]) for t = 1:T]
end


# Cut aggregation logic

# Cluster by peak |gradient| time index.
# Returns Vector{Vector{Int}}: clusters[k] = list of generator indices.
function make_peak_time_clusters(grad_per_gen::Vector{Vector{Float64}}, T::Int, K::Int = 8)
    clusters = [Int[] for _ = 1:K]

    for g in eachindex(grad_per_gen)
        grad = grad_per_gen[g]

        # t* = argmax_t |grad[t]|
        tstar = 1
        best = abs(grad[1])
        @inbounds for t = 2:T
            v = abs(grad[t])
            if v > best
                best = v
                tstar = t
            end
        end

        # Map t* to one of K bins (equal partition of 1..T)
        # bin = ceil(K * tstar / T), but in integer arithmetic:
        b = Int(cld(K * tstar, T))   # cld = ceiling division
        b = min(max(b, 1), K)
        push!(clusters[b], g)
    end

    return clusters
end
