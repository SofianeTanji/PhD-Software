"""A uniform property/oracle propagation rule."""
struct InferenceRule
    id::Symbol
    kind::Symbol
    operation::Union{Nothing,ObjectiveOperation}
    inputs::Vector{Vector{Symbol}}
    output::Symbol
    formula::Symbol
    variadic::Bool

    function InferenceRule(
        id::Symbol,
        kind::Symbol,
        operation::Union{Nothing,ObjectiveOperation},
        inputs,
        output::Symbol,
        formula::Symbol;
        variadic::Bool=false,
    )
        kind in (:property, :oracle) ||
            throw(ArgumentError("an inference rule kind must be :property or :oracle"))
        groups = Vector{Symbol}[Symbol[group...] for group in inputs]
        variadic &&
            length(groups) != 1 &&
            throw(ArgumentError("a variadic inference rule needs one repeated input group"))
        return new(id, kind, operation, groups, output, formula, variadic)
    end
end

"""The collection of rules and named rule formulas used by inference."""
struct RuleSet
    rules::Vector{InferenceRule}
    formulas::Dict{Symbol,Function}
end

function RuleSet(rules=InferenceRule[]; formulas=Dict{Symbol,Function}())
    return RuleSet(InferenceRule[rules...], Dict{Symbol,Function}(formulas))
end

function add_rules(base::RuleSet, rules::InferenceRule...; formulas=Dict{Symbol,Function}())
    merged_formulas = copy(base.formulas)
    for (name, formula) in formulas
        haskey(merged_formulas, name) &&
            throw(ArgumentError("duplicate inference formula: $name"))
        merged_formulas[name] = formula
    end
    identifiers = Set(rule.id for rule in base.rules)
    for rule in rules
        rule.id in identifiers &&
            throw(ArgumentError("duplicate inference rule: $(rule.id)"))
        push!(identifiers, rule.id)
    end
    return RuleSet(vcat(base.rules, InferenceRule[rules...]), merged_formulas)
end

_fact_name(value::Property) = value.name
_fact_name(value::Oracle) = value.name

function _find_fact(facts, name::Symbol)
    for fact in facts
        _fact_name(fact) === name && return fact
    end
    return nothing
end

function _put_fact!(facts::Vector{T}, fact::T) where {T}
    _find_fact(facts, _fact_name(fact)) === nothing && push!(facts, fact)
    return facts
end

function _property_parameter(value::Property, name::Symbol)
    return get(value.parameters, name, nothing)
end

function _primary_property_parameter(value::Property)
    parameter = get(_PROPERTY_PARAMETER, value.name, nothing)
    parameter === nothing && return nothing
    return _property_parameter(value, parameter)
end

function _property_with_primary(name::Symbol, value)
    parameter = get(_PROPERTY_PARAMETER, name, nothing)
    parameter === nothing && return Property(name, Dict{Symbol,ScalarExpr}())
    value === nothing && return Property(name, Dict{Symbol,ScalarExpr}())
    return Property(name, Dict(parameter => scalar(value)))
end

function _rule_qualitative(rule::InferenceRule, term::Term, groups, rules::RuleSet)
    rule.kind === :property && return Property(rule.output, Dict{Symbol,ScalarExpr}())
    dependencies = Oracle[fact for group in groups for fact in group if fact isa Oracle]
    return _combined_oracle(rule.output, dependencies)
end

function _rule_copy_property(rule::InferenceRule, term::Term, groups, rules::RuleSet)
    source = first(first(groups))
    source isa Property || return nothing
    return Property(rule.output, copy(source.parameters))
end

function _rule_linear_smooth(rule::InferenceRule, term::Term, groups, rules::RuleSet)
    return Property(:smooth, Dict(:L => scalar(0)))
end

function _rule_sc_implies_pl(rule::InferenceRule, term::Term, groups, rules::RuleSet)
    source = _find_fact(first(groups), :strongly_convex)
    source === nothing && return nothing
    mu = _property_parameter(source, :mu)
    return _property_with_primary(:polyak_lojasiewicz, mu)
end

function _rule_sum_property(rule::InferenceRule, term::Term, groups, rules::RuleSet)
    values = ScalarExpr[]
    for group in groups
        value = _primary_property_parameter(first(group))
        value === nothing && return Property(rule.output, Dict{Symbol,ScalarExpr}())
        push!(values, value)
    end
    return _property_with_primary(rule.output, reduce(+, values; init=scalar(0)))
end

function _rule_sum_strong_convexity(rule::InferenceRule, term::Term, groups, rules::RuleSet)
    values = ScalarExpr[]
    for child in term.children
        fact = get_property(child, :strongly_convex; rules=rules)
        fact === nothing && continue
        value = _property_parameter(fact, :mu)
        value === nothing && return Property(:strongly_convex, Dict{Symbol,ScalarExpr}())
        push!(values, value)
    end
    isempty(values) && return nothing
    return _property_with_primary(:strongly_convex, reduce(+, values; init=scalar(0)))
end

function _rule_difference_property(rule::InferenceRule, term::Term, groups, rules::RuleSet)
    values = ScalarExpr[]
    for group in groups
        value = _primary_property_parameter(first(group))
        value === nothing && return Property(rule.output, Dict{Symbol,ScalarExpr}())
        push!(values, value)
    end
    return _property_with_primary(rule.output, values[1] + values[2])
end

function _rule_scale_property(rule::InferenceRule, term::Term, groups, rules::RuleSet)
    coefficient = term.coefficient
    coefficient === nothing && return nothing
    source = first(first(groups))
    value = _primary_property_parameter(source)

    if rule.output === :convex
        numeric = try_evaluate_scalar(coefficient)
        numeric === nothing && return nothing
        numeric >= 0 || return nothing
        return Property(:convex, Dict{Symbol,ScalarExpr}())
    end
    if rule.output in (:strongly_convex, :hypo_convex, :centered_quadratic)
        numeric = try_evaluate_scalar(coefficient)
        numeric === nothing && return nothing
        numeric > 0 || return nothing
        return _property_with_primary(
            rule.output, value === nothing ? nothing : coefficient * value
        )
    end
    value === nothing && return Property(rule.output, Dict{Symbol,ScalarExpr}())
    return _property_with_primary(rule.output, abs(coefficient) * value)
end

function _rule_prox_plus_centered_quadratic(
    rule::InferenceRule, term::Term, groups, rules::RuleSet
)
    length(term.children) == 2 || return nothing
    prox_index = findfirst(group -> _find_fact(group, :prox) !== nothing, groups)
    prox_index === nothing && return nothing
    quadratic_index = 3 - prox_index
    centered = get_property(
        term.children[quadratic_index], :centered_quadratic; rules=rules
    )
    centered === nothing && return nothing
    coefficient = _property_parameter(centered, :coefficient)
    coefficient === nothing && return nothing
    known_coefficient = try_evaluate_scalar(coefficient)
    known_coefficient === nothing || known_coefficient > 0 || return nothing
    proximal = _find_fact(groups[prox_index], :prox)
    return _combined_oracle(:prox, Oracle[proximal])
end

function _rule_composition_property(rule::InferenceRule, term::Term, groups, rules::RuleSet)
    outer = first(groups[1])
    inner = first(groups[2])
    if rule.output === :convex
        return Property(:convex, Dict{Symbol,ScalarExpr}())
    elseif rule.output === :linear
        parameters = Dict{Symbol,ScalarExpr}()
        for name in (:operator_norm, :min_singular_value)
            left = _property_parameter(outer, name)
            right = _property_parameter(inner, name)
            left === nothing || right === nothing || (parameters[name] = left * right)
        end
        return Property(:linear, parameters)
    elseif rule.output === :linear_composition
        return Property(:linear_composition, copy(inner.parameters))
    elseif rule.output === :smooth
        L = _property_parameter(outer, :L)
        norm = _property_parameter(inner, :operator_norm)
        L === nothing ||
            norm === nothing ||
            return Property(:smooth, Dict(:L => L * norm^2))
    elseif rule.output === :strongly_convex
        mu = _property_parameter(outer, :mu)
        sigma = _property_parameter(inner, :min_singular_value)
        mu === nothing ||
            sigma === nothing ||
            return Property(:strongly_convex, Dict(:mu => mu * sigma^2))
    elseif rule.output === :lipschitz
        M = _property_parameter(outer, :M)
        scale = if inner.name === :linear
            _property_parameter(inner, :operator_norm)
        else
            _primary_property_parameter(inner)
        end
        M === nothing ||
            scale === nothing ||
            return Property(:lipschitz, Dict(:M => M * scale))
    end
    return nothing
end

function _rule_maximum_property(rule::InferenceRule, term::Term, groups, rules::RuleSet)
    rule.output === :convex && return Property(:convex, Dict{Symbol,ScalarExpr}())
    values = ScalarExpr[]
    for group in groups
        value = _primary_property_parameter(first(group))
        value === nothing && return Property(rule.output, Dict{Symbol,ScalarExpr}())
        push!(values, value)
    end
    result = reduce(max, values)
    return _property_with_primary(rule.output, result)
end

function _combined_exactness(values::Vector{Oracle})
    return _combined_oracle_exactness(values)
end

function _combined_oracle(name::Symbol, dependencies::Vector{Oracle})
    isempty(dependencies) && return Oracle(name)
    costs = ScalarExpr[value.cost for value in dependencies if value.cost !== nothing]
    cost =
        length(costs) == length(dependencies) ? reduce(+, costs; init=scalar(0)) : nothing
    return Oracle(name, _combined_exactness(dependencies), cost)
end

function _rule_combine_oracles(rule::InferenceRule, term::Term, groups, rules::RuleSet)
    dependencies = Oracle[fact for group in groups for fact in group if fact isa Oracle]
    return _combined_oracle(rule.output, dependencies)
end

function _rule_scale_oracle(rule::InferenceRule, term::Term, groups, rules::RuleSet)
    coefficient = term.coefficient
    coefficient === nothing && return nothing
    if rule.output === :prox
        numeric = try_evaluate_scalar(coefficient)
        numeric === nothing && return nothing
        numeric > 0 || return nothing
    end
    return _rule_combine_oracles(rule, term, groups, rules)
end

function _default_inference_formulas()
    return Dict{Symbol,Function}(
        :qualitative          => _rule_qualitative,
        :copy_property        => _rule_copy_property,
        :linear_smooth        => _rule_linear_smooth,
        :sc_implies_pl        => _rule_sc_implies_pl,
        :sum_property         => _rule_sum_property,
        :sum_strong_convexity => _rule_sum_strong_convexity,
        :difference_property  => _rule_difference_property,
        :scale_property       => _rule_scale_property,
        :composition_property => _rule_composition_property,
        :maximum_property     => _rule_maximum_property,
        :combine_oracles      => _rule_combine_oracles,
        :scale_oracle         => _rule_scale_oracle,
        :prox_plus_centered_quadratic => _rule_prox_plus_centered_quadratic,
    )
end

function _default_inference_rules()
    rules = InferenceRule[]
    push!(
        rules,
        InferenceRule(
            :strong_convex_is_convex,
            :property,
            nothing,
            [[:strongly_convex]],
            :convex,
            :qualitative,
        ),
        InferenceRule(
            :linear_is_convex, :property, nothing, [[:linear]], :convex, :qualitative
        ),
        InferenceRule(
            :linear_is_smooth, :property, nothing, [[:linear]], :smooth, :linear_smooth
        ),
        InferenceRule(
            :smooth_sc_is_pl,
            :property,
            nothing,
            [[:smooth, :strongly_convex]],
            :polyak_lojasiewicz,
            :sc_implies_pl,
        ),
        InferenceRule(
            :gradient_is_subgradient,
            :oracle,
            nothing,
            [[:gradient]],
            :subgradient,
            :qualitative,
        ),
    )

    for name in (:convex, :smooth, :lipschitz, :hypo_convex)
        formula = name === :convex ? :qualitative : :sum_property
        push!(
            rules,
            InferenceRule(
                Symbol(:sum_, name),
                :property,
                SumOperation,
                [[name]],
                name,
                formula;
                variadic=true,
            ),
        )
    end
    push!(
        rules,
        InferenceRule(
            :sum_strongly_convex,
            :property,
            SumOperation,
            [[:convex]],
            :strongly_convex,
            :sum_strong_convexity;
            variadic=true,
        ),
    )

    for name in (:smooth, :lipschitz)
        push!(
            rules,
            InferenceRule(
                Symbol(:difference_, name),
                :property,
                DifferenceOperation,
                [[name], [name]],
                name,
                :difference_property,
            ),
        )
    end
    for name in (
        :convex,
        :smooth,
        :strongly_convex,
        :hypo_convex,
        :lipschitz,
        :centered_quadratic,
    )
        push!(
            rules,
            InferenceRule(
                Symbol(:scale_, name),
                :property,
                ScaleOperation,
                [[name]],
                name,
                :scale_property,
            ),
        )
    end
    push!(
        rules,
        InferenceRule(
            :linear_after_linear,
            :property,
            CompositionOperation,
            [[:linear], [:linear]],
            :linear,
            :composition_property,
        ),
        InferenceRule(
            :convex_after_linear,
            :property,
            CompositionOperation,
            [[:convex], [:linear]],
            :convex,
            :composition_property,
        ),
        InferenceRule(
            :smooth_after_linear,
            :property,
            CompositionOperation,
            [[:smooth], [:linear]],
            :smooth,
            :composition_property,
        ),
        InferenceRule(
            :strong_convex_after_linear,
            :property,
            CompositionOperation,
            [[:strongly_convex], [:linear]],
            :strongly_convex,
            :composition_property,
        ),
        InferenceRule(
            :lipschitz_after_linear,
            :property,
            CompositionOperation,
            [[:lipschitz], [:linear]],
            :lipschitz,
            :composition_property,
        ),
        InferenceRule(
            :record_linear_composition,
            :property,
            CompositionOperation,
            [[:convex], [:linear]],
            :linear_composition,
            :composition_property,
        ),
        InferenceRule(
            :maximum_convex,
            :property,
            MaximumOperation,
            [[:convex]],
            :convex,
            :maximum_property;
            variadic=true,
        ),
        InferenceRule(
            :maximum_lipschitz,
            :property,
            MaximumOperation,
            [[:lipschitz]],
            :lipschitz,
            :maximum_property;
            variadic=true,
        ),
    )
    push!(
        rules,
        InferenceRule(
            :prox_plus_centered_quadratic_right,
            :oracle,
            SumOperation,
            [[:prox], Symbol[]],
            :prox,
            :prox_plus_centered_quadratic,
        ),
        InferenceRule(
            :prox_plus_centered_quadratic_left,
            :oracle,
            SumOperation,
            [Symbol[], [:prox]],
            :prox,
            :prox_plus_centered_quadratic,
        ),
    )

    for (operation, label, names) in (
        (SumOperation, :sum, (:value, :gradient, :subgradient)),
        (DifferenceOperation, :difference, (:value, :gradient)),
        (MaximumOperation, :maximum, (:value,)),
    )
        for name in names
            push!(
                rules,
                InferenceRule(
                    Symbol(label, '_', name),
                    :oracle,
                    operation,
                    [[name]],
                    name,
                    :combine_oracles;
                    variadic=true,
                ),
            )
        end
    end
    push!(
        rules,
        InferenceRule(
            :maximum_subgradient,
            :oracle,
            MaximumOperation,
            [[:value, :subgradient]],
            :subgradient,
            :combine_oracles;
            variadic=true,
        ),
    )
    for name in (:value, :gradient, :subgradient, :prox)
        push!(
            rules,
            InferenceRule(
                Symbol(:scale_, name, :_oracle),
                :oracle,
                ScaleOperation,
                [[name]],
                name,
                :scale_oracle,
            ),
        )
    end
    push!(
        rules,
        InferenceRule(
            :composition_value,
            :oracle,
            CompositionOperation,
            [[:value], [:value]],
            :value,
            :combine_oracles,
        ),
        InferenceRule(
            :composition_gradient,
            :oracle,
            CompositionOperation,
            [[:gradient], [:gradient]],
            :gradient,
            :combine_oracles,
        ),
    )
    return rules
end

const DEFAULT_RULES = RuleSet(
    _default_inference_rules(); formulas=_default_inference_formulas()
)

"""Return an independent copy of the built-in inference rules."""
default_rules() = deepcopy(DEFAULT_RULES)

function _direct_facts(value::Term, kind::Symbol)
    kind === :property && return copy(value.properties)
    kind === :oracle && return copy(value.oracles)
    throw(ArgumentError("unknown fact kind: $kind"))
end

function _facts(value::Term, kind::Symbol, rules::RuleSet)
    facts = _direct_facts(value, kind)

    for rule in rules.rules
        rule.kind === kind || continue
        rule.operation === value.operation || continue
        rule.operation === nothing && continue
        groups = Vector{Any}[]
        if rule.variadic
            required = only(rule.inputs)
            for child in value.children
                child_facts = _facts(child, kind, rules)
                selected = Any[]
                for name in required
                    fact = _find_fact(child_facts, name)
                    fact === nothing && (empty!(selected); break)
                    push!(selected, fact)
                end
                isempty(selected) && !isempty(required) && (empty!(groups); break)
                push!(groups, selected)
            end
            isempty(groups) && !isempty(value.children) && continue
        else
            length(rule.inputs) == length(value.children) || continue
            matched = true
            for (child, required) in zip(value.children, rule.inputs)
                child_facts = _facts(child, kind, rules)
                selected = Any[]
                for name in required
                    fact = _find_fact(child_facts, name)
                    if fact === nothing
                        matched = false
                        break
                    end
                    push!(selected, fact)
                end
                matched || break
                push!(groups, selected)
            end
            matched || continue
        end
        formula = get(rules.formulas, rule.formula, nothing)
        formula === nothing &&
            throw(ArgumentError("unknown inference formula: $(rule.formula)"))
        derived = formula(rule, value, groups, rules)
        derived === nothing || _put_fact!(facts, derived)
    end

    changed = true
    while changed
        changed = false
        for rule in rules.rules
            rule.kind === kind || continue
            rule.operation === nothing || continue
            required = only(rule.inputs)
            selected = Any[]
            for name in required
                fact = _find_fact(facts, name)
                fact === nothing && (empty!(selected); break)
                push!(selected, fact)
            end
            isempty(selected) && !isempty(required) && continue
            _find_fact(facts, rule.output) === nothing || continue
            formula = get(rules.formulas, rule.formula, nothing)
            formula === nothing &&
                throw(ArgumentError("unknown inference formula: $(rule.formula)"))
            derived = formula(rule, value, Any[selected], rules)
            derived === nothing && continue
            push!(facts, derived)
            changed = true
        end
    end
    return facts
end

function properties(value::Term; rules::RuleSet=DEFAULT_RULES)
    Property[_facts(value, :property, rules)...]
end
function properties(value::Problem; rules::RuleSet=DEFAULT_RULES)
    properties(value.objective; rules=rules)
end
function oracles(value::Term; rules::RuleSet=DEFAULT_RULES)
    Oracle[_facts(value, :oracle, rules)...]
end
function oracles(value::Problem; rules::RuleSet=DEFAULT_RULES)
    oracles(value.objective; rules=rules)
end

function get_property(value::Term, name::Symbol; rules::RuleSet=DEFAULT_RULES)
    _find_fact(properties(value; rules=rules), name)
end
function get_property(value::Problem, name::Symbol; rules::RuleSet=DEFAULT_RULES)
    get_property(value.objective, name; rules=rules)
end
function has_property(
    value::Union{Term,Problem}, name::Symbol; rules::RuleSet=DEFAULT_RULES
)
    get_property(value, name; rules=rules) !== nothing
end

function get_oracle(value::Term, name::Symbol; rules::RuleSet=DEFAULT_RULES)
    _find_fact(oracles(value; rules=rules), name)
end
function get_oracle(value::Problem, name::Symbol; rules::RuleSet=DEFAULT_RULES)
    get_oracle(value.objective, name; rules=rules)
end
function has_oracle(value::Union{Term,Problem}, name::Symbol; rules::RuleSet=DEFAULT_RULES)
    get_oracle(value, name; rules=rules) !== nothing
end
