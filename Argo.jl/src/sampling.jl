module Sampling

using Random
using ..Argo

export Empirical, Uniform, draw, rank

abstract type AbstractDistribution end

"""A continuous uniform distribution used for symbolic ranking."""
struct Uniform <: AbstractDistribution
    lower::Float64
    upper::Float64

    function Uniform(lower::Real, upper::Real)
        isfinite(lower) && isfinite(upper) && lower <= upper ||
            throw(ArgumentError("uniform endpoints must be finite and ordered"))
        return new(Float64(lower), Float64(upper))
    end
end

"""A finite empirical distribution sampled with replacement."""
struct Empirical <: AbstractDistribution
    values::Vector{Float64}

    function Empirical(values)
        samples = Float64[value for value in values]
        isempty(samples) &&
            throw(ArgumentError("an empirical distribution must not be empty"))
        all(isfinite, samples) || throw(ArgumentError("empirical values must be finite"))
        return new(samples)
    end
end

function draw(rng::AbstractRNG, distribution::Uniform)
    distribution.lower + (distribution.upper - distribution.lower) * rand(rng)
end
draw(rng::AbstractRNG, distribution::Empirical) = rand(rng, distribution.values)
draw(rng::AbstractRNG, value::Real) = value
function draw(rng::AbstractRNG, sampler::Function)
    return applicable(sampler, rng) ? sampler(rng) : sampler()
end

function _sample_values(rng, distributions, fixed_values)
    result = Dict{Any,Any}(pairs(fixed_values))
    for (name, distribution) in pairs(distributions)
        sampled = draw(rng, distribution)
        sampled isa Real ||
            throw(ArgumentError("a sampling distribution must produce real values"))
        isfinite(sampled) ||
            throw(ArgumentError("a sampling distribution produced a nonfinite value"))
        result[name] = sampled
    end
    return result
end

function _bound_score(certificate, iterations, context)
    evaluated = Argo.Authoring.evaluate_bound(certificate, iterations, context)
    return evaluated.status === Argo.Authoring.NumericBound ? evaluated.value : nothing
end

function _complexity_score(certificate, request)
    ranked = Argo.rank(Argo.Certificate[certificate], request)
    isempty(ranked) && return nothing
    return Argo.metric(only(ranked), :oracle_cost).value
end

function _average_rank!(rank_observations, best_credit, scores)
    valid = Int[index for index in eachindex(scores) if scores[index] !== nothing]
    isempty(valid) && return nothing
    sort!(valid; by=index -> scores[index])
    cursor = 1
    while cursor <= length(valid)
        stop = cursor
        while stop < length(valid) && scores[valid[stop + 1]] == scores[valid[cursor]]
            stop += 1
        end
        average_rank = (cursor + stop) / 2
        for position in cursor:stop
            push!(rank_observations[valid[position]], average_rank)
        end
        if cursor == 1
            credit = 1 / (stop - cursor + 1)
            for position in cursor:stop
                best_credit[valid[position]] += credit
            end
        end
        cursor = stop + 1
    end
end

function _mean_and_se(values::Vector{Float64})
    isempty(values) && return Inf, Inf
    mean = sum(values) / length(values)
    length(values) == 1 && return mean, 0.0
    variance = sum((value - mean)^2 for value in values) / (length(values) - 1)
    return mean, sqrt(variance / length(values))
end

"""
    rank(certificates; distributions, samples, seed, iterations=...)
    rank(certificates; distributions, samples, seed, accuracy=...)

Estimate the ordering of symbolic bounds at a fixed iteration, or of symbolic
oracle complexities at a fixed accuracy. Every distribution, sample count, and
seed is explicit.
"""
function rank(
    certificates::AbstractVector{<:Argo.Certificate};
    distributions,
    samples::Integer,
    seed::Integer,
    iterations::Union{Nothing,Integer}=nothing,
    accuracy::Union{Nothing,Real}=nothing,
    initial_bounds=Dict(),
    oracle_costs=Dict(),
    values=Dict(),
    max_iterations::Integer=1_000_000_000,
)
    samples > 0 || throw(ArgumentError("samples must be positive"))
    xor(iterations === nothing, accuracy === nothing) ||
        throw(ArgumentError("provide exactly one of iterations or accuracy"))
    iterations === nothing ||
        iterations >= 0 ||
        throw(ArgumentError("iterations must be nonnegative"))
    rng = MersenneTwister(seed)
    count = length(certificates)
    observations = [Float64[] for _ in 1:count]
    best_credit = zeros(Float64, count)
    for _ in 1:Int(samples)
        sampled_values = _sample_values(rng, distributions, values)
        context = Argo.ComparisonContext(;
            initial_bounds=initial_bounds,
            oracle_costs=oracle_costs,
            values=sampled_values,
            max_iterations=max_iterations,
        )
        scores = Union{Nothing,Float64}[]
        for certificate in certificates
            score = if iterations !== nothing
                _bound_score(certificate, Int(iterations), context)
            else
                _complexity_score(certificate, Argo.RankingRequest(accuracy, context))
            end
            push!(scores, score)
        end
        _average_rank!(observations, best_credit, scores)
    end
    results = Argo.PlanAssessment[]
    for index in eachindex(certificates)
        expected_rank, expected_rank_se = _mean_and_se(observations[index])
        probability = best_credit[index] / samples
        probability_se = sqrt(probability * (1 - probability) / samples)
        push!(
            results,
            Argo.PlanAssessment(
                Argo.PlanPhase[Argo.PlanPhase(
                    certificates[index], iterations === nothing ? nothing : Int(iterations)
                ),],
                nothing,
                Argo.MetricEstimate[
                    Argo.MetricEstimate(:probability_best, probability, probability_se),
                    Argo.MetricEstimate(:expected_rank, expected_rank, expected_rank_se),
                    Argo.MetricEstimate(
                        :valid_observations, Float64(length(observations[index])), nothing
                    ),
                ],
                Dict{Symbol,Argo.ScalarExpr}(),
            ),
        )
    end
    sort!(
        results;
        by=value -> (
            -Argo.metric(value, :probability_best).value,
            Argo.metric(value, :expected_rank).value,
            string(Argo.Authoring.method_declaration(only(value.phases).certificate).id),
        ),
    )
    return results
end

end
