# ═══════════════════════════════════════════════════════════════════════════════
# common.jl — Shared imports, constants, helpers, and method dispatch
# ═══════════════════════════════════════════════════════════════════════════════

using ConvexHullPricing
using DataFrames, Statistics
using JLD2

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

const BUDGET = 15 * 60.0    # 15 min
const BUDGET_5 = 5 * 60.0     # 5 min (early-time table, FGM sensitivity)
const BUDGET_TUNE = 30.0        # 30 s (refined tuning phase)
const TIME_TO_THRESHOLD_FACTOR = 1e-5  # target excess = 1e-5 × optimal side payments

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")

const METHOD_NAMES = [
    "SUBG",
    "SUBG-L",
    "EST-POL",
    "D-Adapt",
    "DoWG",
    "FGM",
    "PC-BLM",
    "PC-BPLM",
    "DLM",
    "CG",
]

const BUNDLE_METHODS = ["PC-BLM", "PC-BPLM", "DLM", "MC-BLM", "MC-BPLM", "BLM", "BPLM"]

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

function load_instances()
    BE = [load_data(f) for f in sort(readdir("data/belgian"; join = true))]
    CA = [load_ca_data(f) for f in sort(readdir("data/ca"; join = true))]
    @info "Loaded $(length(BE)) Belgian + $(length(CA)) Californian instances."
    return BE, CA
end

"""Continuous-relaxation dual prices (warm start)."""
get_x0(instance) = LP_Relaxation(instance)

# ─────────────────────────────────────────────────────────────────────────────
# Save / load helpers
# ─────────────────────────────────────────────────────────────────────────────

function save_run(dir, name, result)
    mkpath(dir)
    path = joinpath(dir, "$name.jld2")
    x_best, iterates, fun_iterates, time_vector = result[1], result[2], result[3], result[4]
    oracle_times = length(result) >= 5 ? result[5] : Float64[]
    jldsave(path; x_best, iterates, fun_iterates, time_vector, oracle_times)
    return path
end

function load_run(path)
    d = load(path)
    base = (d["x_best"], d["iterates"], d["fun_iterates"], d["time_vector"])
    ot = haskey(d, "oracle_times") ? d["oracle_times"] : Float64[]
    return (base..., ot)
end

function save_run_with_bounds(dir, name, result)
    mkpath(dir)
    path = joinpath(dir, "$name.jld2")
    x_best, iterates, fun_iterates, time_vector, oracle_times, UBs, LBs = result
    jldsave(
        path;
        x_best,
        iterates,
        fun_iterates,
        time_vector,
        oracle_times,
        UpperBounds = UBs,
        LowerBounds = LBs,
    )
    return path
end

"""Load all ground-truth L* values as Dict("BE1" => val, ...)."""
function load_ground_truth()
    dir = joinpath(RESULTS_DIR, "ground_truth")
    L = Dict{String,Float64}()
    for f in readdir(dir; join = true)
        endswith(f, ".jld2") || continue
        d = load(f)
        key = replace(splitext(basename(f))[1], "GT_" => "")
        isempty(d["fun_iterates"]) &&
            (@warn "Empty fun_iterates in $f — skipping"; continue)
        L[key] = maximum(d["fun_iterates"])
    end
    return L
end


# ─────────────────────────────────────────────────────────────────────────────
# Analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

"""Zero-order-hold interpolation of (tvec, fvals) onto sorted query grid tgrid."""
function interpolate_onto_grid(tvec, fvals, tgrid)
    out = fill(fvals[1], length(tgrid))
    j = 1
    for (i, tq) in enumerate(tgrid)
        while j < length(tvec) && tvec[j+1] <= tq
            j += 1
        end
        out[i] = tvec[j] <= tq ? fvals[j] : fvals[1]
    end
    return out
end

"""Return true (and log) if path already exists. Use as a loop guard."""
skip_if_exists(path) = isfile(path) ? (@info "Skipping (exists): $path"; true) : false

"""
Best-so-far objective trajectory for a method.

Dual methods maximize their objective, while CG minimizes the primal objective.
"""
best_so_far(method::String, fvals) = method == "CG" ? accumulate(min, fvals) : accumulate(max, fvals)

"""
Best-so-far objective value at a given cutoff time.
"""
function best_so_far_at_cutoff(method::String, fvals, tvec, cutoff::Real)
    bsf = best_so_far(method, fvals)
    return first(interpolate_onto_grid(tvec[2:end], bsf, [cutoff]))
end

"""
Promote a baseline reference value using the best values observed on disk.

For dual methods, a larger best-so-far value tightens the lower side of the feasible
reference interval. For CG, a smaller best-so-far value tightens the upper side.
When the interval is non-empty, the returned reference is the baseline projected
onto that interval.
"""
function promoted_reference_value(
    tag::String,
    baseline::Float64;
    bdir::String = joinpath(RESULTS_DIR, "benchmark"),
    methods = METHOD_NAMES,
)
    lower = baseline
    upper = Inf

    for m in methods
        path = joinpath(bdir, "$(m)_$(tag).jld2")
        isfile(path) || continue
        _, _, fvals, _, _ = load_run(path)
        isempty(fvals) && continue
        candidate = best_so_far(m, fvals)[end]
        if m == "CG"
            upper = min(upper, candidate)
        else
            lower = max(lower, candidate)
        end
    end

    if isfinite(upper) && lower > upper
        @warn "Promoted reference interval is empty for $tag; using strongest dual value." baseline lower upper
        return lower
    end

    return isfinite(upper) ? clamp(baseline, lower, upper) : max(baseline, lower)
end

"""Excess side payment: L* - D_k (absolute duality gap, ≥ 0)."""
excess_side_payment(Lbest, Lstar) = Lstar - Lbest

function time_to_threshold(tvec, fvals, Lstar, thr, method = "")
    bsf = best_so_far(method, fvals)
    for (i, f) in enumerate(bsf)
        # For CG (primal): excess = f - Lstar (how much worse than optimal)
        # For dual methods: excess = Lstar - f (duality gap)
        excess = method == "CG" ? f - Lstar : Lstar - f
        if excess <= thr
            return tvec[min(i + 1, length(tvec))]
        end
    end
    return Inf
end

# ─────────────────────────────────────────────────────────────────────────────
# Instance tags
# ─────────────────────────────────────────────────────────────────────────────

be_tag(idx) = "BE$idx"
ca_tag(idx) = "CA$idx"

n_be() = length(readdir("data/belgian"))
n_ca() = length(readdir("data/ca"))

"""Return all (instance, dataset_label, tag) triples for 3 datasets."""
function all_tagged_instances(BE, CA)
    insts = Tuple[]
    for (i, inst) in enumerate(BE)
        ;
        push!(insts, (inst, "BE", be_tag(i)));
    end
    for (i, inst) in enumerate(CA)
        ;
        push!(insts, (inst, "CA", ca_tag(i)));
    end
    return insts
end

"""All instance tags across all datasets."""
function all_tags()
    vcat([be_tag(i) for i = 1:n_be()], [ca_tag(i) for i = 1:n_ca()])
end

# ─────────────────────────────────────────────────────────────────────────────
# Method dispatch
# ─────────────────────────────────────────────────────────────────────────────

function _get_ground_truth_value(tag)
    path = joinpath(RESULTS_DIR, "ground_truth", "GT_$tag.jld2")
    isfile(path) || return nothing
    d = load(path)
    isempty(d["fun_iterates"]) && return nothing
    return maximum(d["fun_iterates"])
end

"""Lookup dataset-specific hyperparameter for a method (hardcoded fallback)."""
function _ds_param(ds, be_val, ca_val)
    return ds == "BE" ? be_val : ca_val
end

# ─── Tuning results cache ───────────────────────────────────────────────────

"""Cached tuning results: maps (method, dataset) → best hyperparameter."""
const _TUNING_CACHE = Ref{Union{Nothing,Dict{Tuple{String,String},Float64}}}(nothing)

function _load_tuning_cache()
    _TUNING_CACHE[] !== nothing && return _TUNING_CACHE[]
    path = joinpath(RESULTS_DIR, "tuning", "tuning_results.jld2")
    if !isfile(path)
        @info "No tuning results found at $path — using hardcoded defaults."
        _TUNING_CACHE[] = Dict{Tuple{String,String},Float64}()
        return _TUNING_CACHE[]
    end
    df = load(path)["all_results"]
    cache = Dict{Tuple{String,String},Float64}()
    for key in unique(zip(df.method, df.dataset))
        m, ds = key
        rows = filter(r -> r.method == m && r.dataset == ds, df)
        best_idx = argmax(rows.best_obj)
        cache[(m, ds)] = rows.param[best_idx]
    end
    @info "Loaded tuning results for $(length(cache)) (method, dataset) pairs."
    _TUNING_CACHE[] = cache
    return cache
end

"""Look up tuned hyperparameter for `method` on `ds`, falling back to `default`."""
function _tuned_param(method, ds, default)
    cache = _load_tuning_cache()
    return get(cache, (method, ds), default)
end

# ─── SUBG-L calibration cache ───────────────────────────────────────────────

const _SUBGL_CACHE = Ref{Union{Nothing,Dict{String,Any}}}(nothing)

function _load_subgl_cache()
    _SUBGL_CACHE[] !== nothing && return _SUBGL_CACHE[]
    path = joinpath(RESULTS_DIR, "tuning", "subgl_calibration.jld2")
    if !isfile(path)
        @info "No SUBG-L calibration found at $path — using hardcoded defaults."
        _SUBGL_CACHE[] = Dict{String,Any}()
        return _SUBGL_CACHE[]
    end
    _SUBGL_CACHE[] = Dict{String,Any}(load(path))
    @info "Loaded SUBG-L calibration."
    return _SUBGL_CACHE[]
end

function _subgl_N(ds, default)
    cache = _load_subgl_cache()
    return get(cache, "N_$ds", default)
end

# ─────────────────────────────────────────────────────────────────────────────

function run_method(method, inst, x0, budget, ds; lstar = nothing, stop = nothing, kwargs...)
    _stop(b) =
        isnothing(stop) ? (isnothing(lstar) ? TimeBudget(b) : TimeBudgetWithLstar(b, lstar)) :
        stop

    # ── Subgradient family ──
    if method == "SUBG"
        α = _tuned_param("SUBG", ds, _ds_param(ds, 100.0, 0.3))
        return SubgradientMethod(inst, x0, _stop(budget), α; normalize_gradient = true)

    elseif method == "SUBG-L"
        N = _subgl_N(ds, _ds_param(ds, 1200, 400))
        R = _tuned_param("SUBG-L", ds, _ds_param(ds, 40.0, 0.2))
        return LastIterateSubgradientMethod(
            inst,
            x0,
            N,
            R;
            Lstar = isnothing(lstar) ? Inf : lstar,
        )

    elseif method == "EST-POL"
        α = _tuned_param("EST-POL", ds, _ds_param(ds, 32000.0, 300.0))
        return EstimatedPolyak(inst, x0, _stop(budget), α)

    elseif method == "D-Adapt"
        d0 = _tuned_param("D-Adapt", ds, _ds_param(ds, 10.0, 0.15))
        return DAdaptation(inst, x0, _stop(budget), d0)

    elseif method == "DoWG"
        d0 = _tuned_param("DoWG", ds, _ds_param(ds, 20.0, 0.1))
        return DowG(inst, x0, _stop(budget), d0)

        # ── Smoothing ──
    elseif method == "FGM"
        η = _tuned_param("FGM", ds, _ds_param(ds, 1e-4, 1e-5))
        return FastGradientMethod(
            inst,
            x0,
            _stop(budget),
            1e-8;
            stepsize = η,
            use_shift = true,
            recompute_exact = true,
        )

        # ── Bundle level methods ──
    elseif method == "BLM"
        α = _tuned_param("BLM", ds, 0.9)
        ub = fast_uc_upper_bound(inst; time_limit = 10.0)
        return BundleLevelMethod(
            inst,
            x0,
            _stop(budget),
            α;
            initial_ub_lstar = ub,
            kwargs...,
        )

    elseif method == "BPLM"
        α = _tuned_param("BPLM", ds, _ds_param(ds, 0.95, 0.9))
        ub = fast_uc_upper_bound(inst; time_limit = 10.0)
        return BundleProximalLevelMethod(
            inst,
            x0,
            _stop(budget),
            α;
            initial_ub_lstar = ub,
            kwargs...,
        )

    elseif method == "DLM"
        α = _tuned_param("DLM", ds, 0.5)
        ub = fast_uc_upper_bound(inst; time_limit = 10.0)
        return DynamicLevelMethod(
            inst,
            x0,
            _stop(budget),
            α;
            initial_ub_lstar = ub,
            kwargs...,
        )

        # ── Multi-cut bundle methods ──
    elseif method == "MC-BLM"
        α = _tuned_param("MC-BLM", ds, 0.3)
        ub = fast_uc_upper_bound(inst; time_limit = 10.0)
        return MulticutBundleLevelMethod(
            inst,
            x0,
            _stop(budget),
            α;
            initial_ub_lstar = ub,
            kwargs...,
        )

    elseif method == "MC-BPLM"
        α = _tuned_param("MC-BPLM", ds, 0.3)
        ub = fast_uc_upper_bound(inst; time_limit = 10.0)
        return MulticutBundleProximalLevelMethod(
            inst,
            x0,
            _stop(budget),
            α;
            initial_ub_lstar = ub,
            kwargs...,
        )

    elseif method == "PC-BLM"
        α = 0.9
        ub = fast_uc_upper_bound(inst; time_limit = 10.0)
        return PreconditionedLevelMethod(
            inst,
            x0,
            _stop(budget),
            α;
            initial_ub_lstar = ub,
            kwargs...,
        )

    elseif method == "PC-BPLM"
        α = 0.9
        ub = fast_uc_upper_bound(inst; time_limit = 10.0)
        return PreconditionedProximalLevelMethod(
            inst,
            x0,
            _stop(budget),
            α;
            initial_ub_lstar = ub,
            kwargs...,
        )

        # ── Primal methods ──
    elseif method == "CG"
        eps = _tuned_param("CG", ds, 1e-6)
        return ColumnGeneration(inst, x0, TimeBudget(budget), eps)

    else
        error("Unknown method: $method")
    end
end

function run_method_polyak(inst, x0, budget, Lstar)
    return PolyakMethod(inst, x0, TimeBudget(budget), Lstar)
end
