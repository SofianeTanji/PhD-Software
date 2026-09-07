module Hybrid

using ..Argo

export ExhaustiveSwitchSearch,
    UnimodalSwitchSearch,
    certificates,
    optimize

struct ExhaustiveSwitchSearch end

struct UnimodalSwitchSearch
    provenance::String

    function UnimodalSwitchSearch(provenance::AbstractString)
        isempty(provenance) &&
            throw(ArgumentError("unimodal switch search needs explicit provenance"))
        return new(String(provenance))
    end
end

function _compatible(first::Argo.Certificate, second::Argo.Certificate)
    Argo.Authoring.same_formulation(first, second) || return false
    Argo.Authoring.method_declaration(first).id ===
    Argo.Authoring.method_declaration(second).id && return false
    second_initial = Argo.Authoring.initial_bound_quantity(second)
    second_initial === :none && return false
    return Argo.Authoring.requested_quantity(first) === second_initial
end

function _phase_context(context::Argo.ComparisonContext, name::Symbol, value::Real)
    result = copy(context.initial_bounds)
    result[name] = value
    return Argo.with(context; initial_bounds=result)
end

function _prefix_work(prefix::Symbol, work::Argo.Authoring.OracleWork)
    return Argo.Authoring.prefix_oracle_work(work, prefix)
end

function _combined_complexity(
    first, second, first_iterations, second_iterations, first_context, second_context
)
    first_complexity = Argo.oracle_complexity(first, first_iterations, first_context)
    first_complexity === nothing && return nothing
    second_complexity = Argo.oracle_complexity(second, second_iterations, second_context)
    second_complexity === nothing && return nothing
    work = Argo.Authoring.OracleWork[]
    append!(work, (_prefix_work(:phase1, value) for value in first_complexity.calls))
    append!(work, (_prefix_work(:phase2, value) for value in second_complexity.calls))
    return Argo.OracleComplexity(
        first_iterations + second_iterations, Argo.Authoring.merge_oracle_work(work)
    )
end

function _automatic_switch_search(first::Argo.Certificate, second::Argo.Certificate)
    if first.theorem.id === :taylor2017_simplified_apg &&
       second.theorem.id === :fista_composite_sc_gap_from_gap_derived
        return UnimodalSwitchSearch(
            "For SAPG followed by SC-FISTA, log B(t) has one stationary crossing because its rational derivative decreases through the constant log-contraction slope",
        )
    end
    return ExhaustiveSwitchSearch()
end

function _assessment_assignments(context::Argo.ComparisonContext)
    return Dict{Symbol,Argo.ScalarExpr}(
        name => Argo.scalar(value) for (name, value) in context.values if
        name isa Symbol && (value isa Real || value isa Argo.ScalarExpr)
    )
end

function _fixed_candidate(
    first::Argo.Certificate,
    second::Argo.Certificate,
    request::Argo.FixedBudgetRequest,
    switch::Int,
)
    budget = request.iterations
    0 <= switch <= budget || throw(DomainError(switch, "switch must lie in the budget"))
    second_iterations = budget - switch
    phase_one_bound = nothing
    second_context = request.context
    final_bound = if switch == 0
        evaluated = Argo.Authoring.evaluate_bound(second, budget, request.context)
        evaluated.status === Argo.Authoring.NumericBound || return nothing
        evaluated.value
    elseif switch == budget
        Argo.Authoring.requested_quantity(first) ===
            Argo.Authoring.requested_quantity(second) || return nothing
        evaluated = Argo.Authoring.evaluate_bound(first, budget, request.context)
        evaluated.status === Argo.Authoring.NumericBound || return nothing
        phase_one_bound = evaluated.value
        evaluated.value
    else
        evaluated = Argo.Authoring.evaluate_bound(first, switch, request.context)
        evaluated.status === Argo.Authoring.NumericBound || return nothing
        phase_one_bound = evaluated.value
        second_context = _phase_context(
            request.context,
            Argo.Authoring.initial_bound_quantity(second),
            phase_one_bound,
        )
        final = Argo.Authoring.evaluate_bound(second, second_iterations, second_context)
        final.status === Argo.Authoring.NumericBound || return nothing
        final.value
    end
    complexity = _combined_complexity(
        first,
        second,
        switch,
        second_iterations,
        request.context,
        second_context,
    )
    metrics = Argo.MetricEstimate[
        Argo.MetricEstimate(:bound, final_bound, nothing),
        Argo.MetricEstimate(:switch_time, Float64(switch), nothing),
    ]
    phase_one_bound === nothing || push!(
        metrics, Argo.MetricEstimate(:phase_one_bound, phase_one_bound, nothing)
    )
    if complexity !== nothing
        expression = Argo.Authoring.complexity_cost(complexity, request.context)
        cost = Argo.try_evaluate_scalar(expression, request.context.values)
        cost === nothing ||
            (isfinite(cost) && cost >= 0) &&
            push!(metrics, Argo.MetricEstimate(:oracle_cost, Float64(cost), nothing))
    end
    return Argo.PlanAssessment(
        Argo.PlanPhase[
            Argo.PlanPhase(first, switch),
            Argo.PlanPhase(second, second_iterations),
        ],
        complexity,
        metrics,
        _assessment_assignments(request.context),
    )
end

function _with_switch_evaluations(assessment::Argo.PlanAssessment, count::Int)
    metrics = copy(assessment.metrics)
    push!(metrics, Argo.MetricEstimate(:switch_evaluations, Float64(count), nothing))
    return Argo.PlanAssessment(
        assessment.phases,
        assessment.complexity,
        metrics,
        assessment.assignments,
    )
end

function _best_fixed(candidates)
    valid = Argo.PlanAssessment[value for value in candidates if value !== nothing]
    isempty(valid) && return nothing
    return argmin(
        value -> (
            Argo.metric(value, :bound).value,
            something(value.phases[1].iterations),
        ),
        valid,
    )
end

"""Minimize a compatible chained bound at one fixed total budget."""
function optimize(
    first::Argo.Certificate,
    second::Argo.Certificate,
    request::Argo.FixedBudgetRequest;
    search::Union{Nothing,ExhaustiveSwitchSearch,UnimodalSwitchSearch}=nothing,
    include_endpoints::Bool=true,
)
    _compatible(first, second) || return nothing
    selected_search = search === nothing ? _automatic_switch_search(first, second) : search
    budget = request.iterations
    cache = Dict{Int,Union{Nothing,Argo.PlanAssessment}}()
    evaluate_at(switch) = get!(cache, switch) do
        _fixed_candidate(first, second, request, switch)
    end

    candidates = Union{Nothing,Argo.PlanAssessment}[]
    if include_endpoints
        push!(candidates, evaluate_at(0))
        budget == 0 || push!(candidates, evaluate_at(budget))
    end
    if budget >= 2
        if selected_search isa ExhaustiveSwitchSearch
            append!(candidates, (evaluate_at(switch) for switch in 1:(budget - 1)))
        else
            left = 1
            right = budget - 1
            fallback = false
            while left < right
                middle = left + (right - left) ÷ 2
                current = evaluate_at(middle)
                following = evaluate_at(middle + 1)
                if current === nothing || following === nothing
                    fallback = true
                    break
                end
                if Argo.metric(current, :bound).value <=
                   Argo.metric(following, :bound).value
                    right = middle
                else
                    left = middle + 1
                end
            end
            if fallback
                append!(candidates, (evaluate_at(switch) for switch in 1:(budget - 1)))
            else
                for switch in max(1, left - 1):min(budget - 1, left + 1)
                    push!(candidates, evaluate_at(switch))
                end
            end
        end
    end
    best = _best_fixed(candidates)
    best === nothing && return nothing
    return _with_switch_evaluations(best, length(cache))
end

"""Optimize the integer switching time for one compatible ordered pair."""
function optimize(
    first::Argo.Certificate,
    second::Argo.Certificate,
    request::Argo.RankingRequest;
    max_switch::Integer=10_000,
)
    _compatible(first, second) || return nothing
    max_switch > 0 || throw(ArgumentError("max_switch must be positive"))
    solo_iterations = Argo.iterations_to_accuracy(first, request)
    solo_iterations === nothing && return nothing
    limit = min(Int(max_switch), max(1, solo_iterations))
    best = nothing
    for switch in 1:limit
        evaluated = Argo.Authoring.evaluate_bound(first, switch, request.context)
        evaluated.status === Argo.Authoring.NumericBound || continue
        phase_one_bound = evaluated.value
        second_context = _phase_context(
            request.context, Argo.Authoring.initial_bound_quantity(second), phase_one_bound
        )
        second_request = Argo.RankingRequest(request.accuracy, second_context)
        second_iterations = Argo.iterations_to_accuracy(second, second_request)
        second_iterations === nothing && continue
        second_iterations > 0 || continue
        complexity = _combined_complexity(
            first, second, switch, second_iterations, request.context, second_context
        )
        complexity === nothing && continue
        cost_expression = Argo.Authoring.complexity_cost(complexity, request.context)
        numeric_cost = Argo.try_evaluate_scalar(cost_expression, request.context.values)
        numeric_cost isa Real || continue
        cost = Float64(numeric_cost)
        isfinite(cost) && cost >= 0 || continue
        candidate = Argo.PlanAssessment(
            Argo.PlanPhase[
                Argo.PlanPhase(first, switch), Argo.PlanPhase(second, second_iterations)
            ],
            complexity,
            Argo.MetricEstimate[
                Argo.MetricEstimate(:oracle_cost, cost, nothing),
                Argo.MetricEstimate(:phase_one_bound, phase_one_bound, nothing),
            ],
            Dict{Symbol,Argo.ScalarExpr}(),
        )
        if best === nothing
            best = candidate
            continue
        end
        candidate_key = (
            Argo.metric(candidate, :oracle_cost).value,
            candidate.phases[1].iterations,
            candidate.phases[2].iterations,
        )
        best_key = (
            Argo.metric(best, :oracle_cost).value,
            best.phases[1].iterations,
            best.phases[2].iterations,
        )
        candidate_key < best_key && (best = candidate)
    end
    return best
end

function optimize(
    first::Argo.Certificate,
    second::Argo.Certificate;
    accuracy::Union{Nothing,Real}=nothing,
    iterations::Union{Nothing,Integer}=nothing,
    initial_bounds=Dict(),
    oracle_costs=Dict(),
    values=Dict(),
    max_switch::Integer=10_000,
    max_iterations::Integer=1_000_000_000,
    search::Union{Nothing,ExhaustiveSwitchSearch,UnimodalSwitchSearch}=nothing,
    include_endpoints::Bool=true,
)
    (accuracy === nothing) == (iterations === nothing) && throw(
        ArgumentError("provide exactly one of accuracy or iterations"),
    )
    if iterations !== nothing
        request = Argo.FixedBudgetRequest(;
            iterations=iterations,
            initial_bounds=initial_bounds,
            oracle_costs=oracle_costs,
            values=values,
        )
        return optimize(
            first,
            second,
            request;
            search=search,
            include_endpoints=include_endpoints,
        )
    end
    request = Argo.RankingRequest(;
        accuracy=accuracy,
        initial_bounds=initial_bounds,
        oracle_costs=oracle_costs,
        values=values,
        max_iterations=max_iterations,
    )
    return optimize(first, second, request; max_switch=max_switch)
end

"""Discover and optimize every compatible two-phase plan for a problem."""
function certificates(
    problem::Argo.Problem,
    ranking::Argo.RankingRequest,
    applicability::Argo.ApplicabilityRequest=Argo.ApplicabilityRequest();
    max_switch::Integer=10_000,
)
    second_phase = Argo.certificates(problem, applicability)
    initial_quantities = unique(
        Symbol[
            Argo.Authoring.initial_bound_quantity(certificate) for
            certificate in second_phase if
            Argo.Authoring.initial_bound_quantity(certificate) !== :none
        ],
    )
    first_phase = Argo.Certificate[]
    for initial_quantity in initial_quantities
        append!(
            first_phase,
            Argo.certificates(problem, Argo.with(applicability; quantity=initial_quantity)),
        )
    end
    result = Argo.PlanAssessment[]
    for first in first_phase, second in second_phase
        candidate = optimize(first, second, ranking; max_switch=max_switch)
        candidate === nothing || push!(result, candidate)
    end
    sort!(
        result;
        by=value -> (
            Argo.metric(value, :oracle_cost).value,
            string(Argo.Authoring.method_declaration(value.phases[1].certificate).id),
            string(Argo.Authoring.method_declaration(value.phases[2].certificate).id),
        ),
    )
    return result
end

"""Discover pure and compatible two-phase plans at one fixed total budget."""
function certificates(
    problem::Argo.Problem,
    budget::Argo.FixedBudgetRequest,
    applicability::Argo.ApplicabilityRequest=Argo.ApplicabilityRequest();
    search::Union{Nothing,ExhaustiveSwitchSearch,UnimodalSwitchSearch}=nothing,
)
    second_phase = Argo.certificates(problem, applicability)
    initial_quantities = unique(
        Symbol[
            Argo.Authoring.initial_bound_quantity(certificate) for
            certificate in second_phase if
            Argo.Authoring.initial_bound_quantity(certificate) !== :none
        ],
    )
    first_phase = Argo.Certificate[]
    for initial_quantity in initial_quantities
        append!(
            first_phase,
            Argo.certificates(problem, Argo.with(applicability; quantity=initial_quantity)),
        )
    end
    result = Argo.rank(second_phase, budget)
    for first in first_phase, second in second_phase
        candidate = optimize(
            first,
            second,
            budget;
            search=search,
            include_endpoints=false,
        )
        candidate === nothing || push!(result, candidate)
    end
    sort!(
        result;
        by=value -> (
            Argo.metric(value, :bound).value,
            length(value.phases),
            string(Argo.Authoring.method_declaration(value.phases[1].certificate).id),
        ),
    )
    return result
end

function certificates(
    problem::Argo.Problem;
    quantity::Symbol=:objective_gap,
    accuracy::Union{Nothing,Real}=nothing,
    iterations::Union{Nothing,Integer}=nothing,
    initial_bounds=Dict(),
    oracle_costs=Dict(),
    values=Dict(),
    max_switch::Integer=10_000,
    max_iterations::Integer=1_000_000_000,
    catalogue::Argo.Authoring.Catalogue=Argo.Authoring.default_catalogue(),
    resolver::Argo.Authoring.AbstractOracleResolver=Argo.Authoring.DirectOracleResolver(),
    rules::Argo.Authoring.RuleSet=Argo.Authoring.default_rules(),
    reformulations=Argo.Authoring.default_reformulations(),
    reformulation_depth::Integer=1,
    max_bindings::Integer=4096,
    search::Union{Nothing,ExhaustiveSwitchSearch,UnimodalSwitchSearch}=nothing,
)
    (accuracy === nothing) == (iterations === nothing) && throw(
        ArgumentError("provide exactly one of accuracy or iterations"),
    )
    applicability = Argo.ApplicabilityRequest(;
        quantity=quantity,
        catalogue=catalogue,
        resolver=resolver,
        rules=rules,
        reformulations=reformulations,
        reformulation_depth=reformulation_depth,
        max_bindings=max_bindings,
    )
    if iterations !== nothing
        budget = Argo.FixedBudgetRequest(;
            iterations=iterations,
            initial_bounds=initial_bounds,
            oracle_costs=oracle_costs,
            values=values,
        )
        return certificates(problem, budget, applicability; search=search)
    end
    ranking = Argo.RankingRequest(;
        accuracy=accuracy,
        initial_bounds=initial_bounds,
        oracle_costs=oracle_costs,
        values=values,
        max_iterations=max_iterations,
    )
    return certificates(problem, ranking, applicability; max_switch=max_switch)
end

end
