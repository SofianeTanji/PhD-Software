module RegimeAnalysis

using ..Argo

export RegimeAxis, RegimeCell, RegimeGrid, regime_grid

"""One numerical comparison axis and its ordered coordinate values."""
struct RegimeAxis
    name::Symbol
    values::Vector{Float64}

    function RegimeAxis(name::Symbol, values::AbstractVector{<:Real})
        isempty(values) && throw(ArgumentError("a regime axis must not be empty"))
        numeric = Float64[values...]
        all(isfinite, numeric) ||
            throw(ArgumentError("regime-axis values must be finite"))
        length(numeric) == length(unique(numeric)) ||
            throw(ArgumentError("regime-axis values must be unique"))
        return new(name, numeric)
    end
end

"""All comparable plans and tolerance-aware winners at one grid coordinate."""
struct RegimeCell
    coordinates::Dict{Symbol,Float64}
    assessments::Vector{Argo.PlanAssessment}
    winners::Vector{Argo.PlanAssessment}
    unavailable::Dict{String,String}
end

"""Renderer-independent one- or two-axis regime data."""
struct RegimeGrid
    axes::Vector{RegimeAxis}
    shape::Vector{Int}
    mode::Symbol
    metric::Symbol
    absolute_tolerance::Float64
    relative_tolerance::Float64
    cells::Vector{RegimeCell}
end

function Base.getindex(grid::RegimeGrid, indices::Integer...)
    length(indices) == length(grid.shape) ||
        throw(ArgumentError("expected $(length(grid.shape)) grid indices"))
    linear = LinearIndices(Tuple(grid.shape))[indices...]
    return grid.cells[linear]
end

function _plan_key(certificate::Argo.Certificate)
    roles = join(
        (
            string(
                role,
                '=',
                join(
                    getfield.(Argo._flatten_objective(term), :scope),
                    '+',
                ),
            ) for (role, term) in Argo.Authoring.role_binding(certificate).roles
        ),
        ';',
    )
    return string(certificate.theorem.id, '|', roles)
end

function _metric_value(assessment::Argo.PlanAssessment, name::Symbol)
    index = findfirst(metric -> metric.name === name, assessment.metrics)
    index === nothing && return nothing
    return assessment.metrics[index].value
end

function _point_request(
    request::Union{Argo.RankingRequest,Argo.FixedBudgetRequest}, coordinates
)
    values = Dict{Any,Any}(pairs(request.context.values))
    merge!(values, coordinates)
    return Argo.with(request; values=values)
end

function _free_parameters(certificate, request)
    k = request isa Argo.FixedBudgetRequest ? request.iterations : 1
    parameters = Argo.ParameterOptimization.declared_parameters(
        certificate;
        k=k,
        initial_bounds=request.context.initial_bounds,
        values=request.context.values,
    )
    return Argo.ParameterOptimization.ScalarParameter[
        parameter for parameter in parameters if
        !haskey(request.context.values, parameter.name)
    ]
end

function _assessment(
    certificate::Argo.Certificate,
    request::Union{Argo.RankingRequest,Argo.FixedBudgetRequest},
    strategies::Argo.ParameterOptimization.StrategyRegistry,
    optimize_parameters::Bool,
)
    parameters = optimize_parameters ? _free_parameters(certificate, request) :
                 Argo.ParameterOptimization.ScalarParameter[]
    if isempty(parameters)
        ranked = Argo.rank(Argo.Certificate[certificate], request)
        return isempty(ranked) ? nothing : only(ranked)
    end
    result = Argo.ParameterOptimization.optimize(
        Argo.Certificate[certificate], parameters, request; strategies=strategies
    )
    return Argo.ParameterOptimization.isoptimized(result) ? result.assessment : nothing
end

function _cell(
    certificates,
    request,
    coordinates,
    metric_name,
    absolute_tolerance,
    relative_tolerance,
    strategies,
    optimize_parameters,
)
    assessments = Argo.PlanAssessment[]
    unavailable = Dict{String,String}()
    for certificate in certificates
        assessment = try
            _assessment(certificate, request, strategies, optimize_parameters)
        catch error
            unavailable[_plan_key(certificate)] = sprint(showerror, error)
            continue
        end
        if assessment === nothing
            unavailable[_plan_key(certificate)] =
                "no numerical assessment or justified parameter strategy"
            continue
        end
        value = _metric_value(assessment, metric_name)
        if value === nothing
            unavailable[_plan_key(certificate)] = "metric $metric_name is unavailable"
            continue
        end
        push!(assessments, assessment)
    end
    sort!(
        assessments;
        by=assessment -> (
            something(_metric_value(assessment, metric_name)),
            _plan_key(only(assessment.phases).certificate),
        ),
    )
    winners = Argo.PlanAssessment[]
    if !isempty(assessments)
        best = something(_metric_value(first(assessments), metric_name))
        for assessment in assessments
            value = something(_metric_value(assessment, metric_name))
            tolerance = absolute_tolerance +
                        relative_tolerance * max(abs(best), abs(value))
            value - best <= tolerance && push!(winners, assessment)
        end
    end
    return RegimeCell(
        Dict{Symbol,Float64}(coordinates), assessments, winners, unavailable
    )
end

"""
Compute generic one- or two-axis regime data. Plotting is deliberately outside
Argo; cells retain all assessments, tolerance-aware ties, and unavailable-plan
reasons for downstream renderers.
"""
function regime_grid(
    certificates::AbstractVector{<:Argo.Certificate},
    request::Union{Argo.RankingRequest,Argo.FixedBudgetRequest},
    axes::RegimeAxis...;
    metric::Symbol=request isa Argo.RankingRequest ? :oracle_cost : :bound,
    absolute_tolerance::Real=0.0,
    relative_tolerance::Real=sqrt(eps(Float64)),
    strategies::Argo.ParameterOptimization.StrategyRegistry=Argo.ParameterOptimization.default_strategies(),
    optimize_parameters::Bool=true,
)
    1 <= length(axes) <= 2 ||
        throw(ArgumentError("a regime grid requires one or two axes"))
    axis_names = getfield.(axes, :name)
    length(axis_names) == length(unique(axis_names)) ||
        throw(ArgumentError("regime-axis names must be unique"))
    isfinite(absolute_tolerance) && absolute_tolerance >= 0 || throw(
        ArgumentError("absolute_tolerance must be finite and nonnegative"),
    )
    isfinite(relative_tolerance) && relative_tolerance >= 0 || throw(
        ArgumentError("relative_tolerance must be finite and nonnegative"),
    )
    shape = length.(getfield.(axes, :values))
    cells = RegimeCell[]
    ranges = map(axis -> eachindex(axis.values), axes)
    for indices in Iterators.product(ranges...)
        coordinates = Dict{Symbol,Float64}(
            axis.name => axis.values[index] for (axis, index) in zip(axes, indices)
        )
        point_request = _point_request(request, coordinates)
        push!(
            cells,
            _cell(
                certificates,
                point_request,
                coordinates,
                metric,
                Float64(absolute_tolerance),
                Float64(relative_tolerance),
                strategies,
                optimize_parameters,
            ),
        )
    end
    mode = request isa Argo.RankingRequest ? :target_accuracy : :fixed_budget
    return RegimeGrid(
        RegimeAxis[axes...],
        Int[shape...],
        mode,
        metric,
        Float64(absolute_tolerance),
        Float64(relative_tolerance),
        cells,
    )
end

end
