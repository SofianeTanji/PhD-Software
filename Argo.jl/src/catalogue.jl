using TOML

"""The location of one catalogue declaration in its tracked TOML source."""
struct SourceLocation
    file::String
    entry::Int
end

"""A validation problem tied to an exact catalogue declaration."""
struct CatalogueIssue
    severity::Symbol
    source::SourceLocation
    field::String
    message::String
end

struct InvalidCatalogue <: Exception
    issues::Vector{CatalogueIssue}
end

function Base.showerror(io::IO, error::InvalidCatalogue)
    println(io, "invalid Argo catalogue:")
    for issue in error.issues
        print(
            io,
            "  ",
            issue.source.file,
            ':',
            issue.source.entry,
            " [",
            issue.field,
            "] ",
            issue.message,
            '\n',
        )
    end
end

"""The certified inner problem needed to realize a controlled-inexact oracle."""
struct InnerSolveRequirement
    quantity::Symbol
    initial_bound::Symbol
    initial_value::ScalarExpr
    max_iterations::Int
end

"""A typed accuracy contract attached to a controlled-inexact oracle."""
struct ControlledInexactness
    model::Symbol
    accuracy::Symbol
    kappa::Symbol
    inner::InnerSolveRequirement
end

"""A theorem-side requirement for an operation on a bound objective role."""
struct OracleRequirement
    name::Symbol
    exactness::OracleExactness
    inexactness::Union{Nothing,ControlledInexactness}

    function OracleRequirement(
        name::Symbol,
        exactness::OracleExactness,
        inexactness::Union{Nothing,ControlledInexactness}=nothing,
    )
        if exactness === ControlledInexactOracle
            inexactness === nothing && throw(
                ArgumentError(
                    "a controlled-inexact requirement needs an accuracy contract"
                ),
            )
        elseif inexactness !== nothing
            throw(
                ArgumentError(
                    "an accuracy contract is only valid for a controlled-inexact requirement",
                ),
            )
        end
        return new(name, exactness, inexactness)
    end
end

function OracleRequirement(
    name::Symbol,
    exactness::Symbol,
    inexactness::Union{Nothing,ControlledInexactness}=nothing,
)
    return OracleRequirement(name, oracle_exactness(exactness), inexactness)
end

"""The properties and oracles required from one objective role."""
struct RoleRequirement
    name::Symbol
    properties::Vector{Symbol}
    oracles::Vector{OracleRequirement}
end

"""The theorem role, property, and parameter that introduced one formula input."""
struct FormulaInputSource
    role::Symbol
    property::Symbol
    parameter::Symbol
end

"""Catalogue data describing a method independently of any convergence theorem."""
struct OracleCall
    role::Symbol
    oracle::Symbol
    count::Int
end

struct MethodDeclaration
    id::Symbol
    name::String
    parameter_vocabulary::Vector{Symbol}
    required_parameters::Vector{Symbol}
    calls::Vector{OracleCall}
    source::SourceLocation
end

function MethodDeclaration(
    id::Symbol,
    name::String,
    parameter_vocabulary::Vector{Symbol},
    calls::Vector{OracleCall},
    source::SourceLocation,
)
    return MethodDeclaration(id, name, parameter_vocabulary, Symbol[], calls, source)
end

"""A theorem-declared admissible interval for one free method parameter."""
struct ParameterDomain
    lower::Union{Float64,Symbol,ScalarExpr}
    upper::Union{Float64,Symbol,ScalarExpr}
    lower_closed::Bool
    upper_closed::Bool
    scale::Symbol

    function ParameterDomain(
        lower::Union{Real,Symbol,ScalarExpr},
        upper::Union{Real,Symbol,ScalarExpr};
        lower_closed::Bool=true,
        upper_closed::Bool=true,
        scale::Symbol=:linear,
    )
        scale in (:linear, :log) ||
            throw(ArgumentError("parameter-domain scale must be :linear or :log"))
        normalized_lower = lower isa Real ? Float64(lower) : lower
        normalized_upper = upper isa Real ? Float64(upper) : upper
        known_lower = normalized_lower isa ScalarExpr ?
                      try_evaluate_scalar(normalized_lower) : normalized_lower
        known_upper = normalized_upper isa ScalarExpr ?
                      try_evaluate_scalar(normalized_upper) : normalized_upper
        if known_lower isa Real
            isfinite(known_lower) ||
                throw(ArgumentError("parameter-domain lower endpoint must be finite"))
            scale === :log && known_lower <= 0 && throw(
                ArgumentError(
                    "a log-scaled parameter domain must have a positive lower endpoint"
                ),
            )
        end
        if known_upper isa Real
            isfinite(known_upper) ||
                throw(ArgumentError("parameter-domain upper endpoint must be finite"))
        end
        if known_lower isa Real && known_upper isa Real
            known_lower <= known_upper ||
                throw(ArgumentError("parameter-domain endpoints must be ordered"))
            known_lower == known_upper &&
                !(lower_closed && upper_closed) && throw(
                ArgumentError("an open parameter domain cannot have equal endpoints"),
            )
        end
        return new(
            normalized_lower,
            normalized_upper,
            lower_closed,
            upper_closed,
            scale,
        )
    end
end

function Base.:(==)(left::ParameterDomain, right::ParameterDomain)
    return left.lower == right.lower &&
           left.upper == right.upper &&
           left.lower_closed == right.lower_closed &&
           left.upper_closed == right.upper_closed &&
           left.scale === right.scale
end


Base.isequal(left::ParameterDomain, right::ParameterDomain) =
    isequal(left.lower, right.lower) &&
    isequal(left.upper, right.upper) &&
    left.lower_closed == right.lower_closed &&
    left.upper_closed == right.upper_closed &&
    left.scale === right.scale

function Base.hash(value::ParameterDomain, seed::UInt)
    return hash(
        value.scale,
        hash(
            value.upper_closed,
            hash(
                value.lower_closed,
                hash(value.upper, hash(value.lower, seed)),
            ),
        ),
    )
end

"""Catalogue data describing one certificate-producing theorem."""
struct TheoremDeclaration
    id::Symbol
    method::Symbol
    name::String
    roles::Vector{RoleRequirement}
    quantity::Symbol
    output::Symbol
    initial_bound::Symbol
    bound::Symbol
    parameter_rules::Dict{Symbol,Symbol}
    parameter_domains::Dict{Symbol,ParameterDomain}
    citation::String
    tightness::Symbol
    notes::Dict{String,Any}
    source::SourceLocation
end

function TheoremDeclaration(
    id::Symbol,
    method::Symbol,
    name::String,
    roles::Vector{RoleRequirement},
    quantity::Symbol,
    output::Symbol,
    initial_bound::Symbol,
    bound::Symbol,
    parameter_rules::Dict{Symbol,Symbol},
    citation::String,
    tightness::Symbol,
    notes::Dict{String,Any},
    source::SourceLocation,
)
    return TheoremDeclaration(
        id,
        method,
        name,
        roles,
        quantity,
        output,
        initial_bound,
        bound,
        parameter_rules,
        Dict{Symbol,ParameterDomain}(),
        citation,
        tightness,
        notes,
        source,
    )
end

"""Named Julia kernels referenced by TOML theorem declarations."""
struct FormulaRegistry
    formulas::Dict{Symbol,Function}
    custom::Set{Symbol}
end

FormulaRegistry() = FormulaRegistry(Dict{Symbol,Function}(), Set{Symbol}())
function FormulaRegistry(formulas::Dict{Symbol,Function})
    return FormulaRegistry(formulas, Set(keys(formulas)))
end
function FormulaRegistry(pairs::Pair{Symbol,<:Function}...)
    formulas = Dict{Symbol,Function}()
    for (name, implementation) in pairs
        haskey(formulas, name) && throw(ArgumentError("duplicate formula $name"))
        formulas[name] = implementation
    end
    return FormulaRegistry(formulas, Set(keys(formulas)))
end

function add_formulas(registry::FormulaRegistry, pairs::Pair{Symbol,<:Function}...)
    formulas = copy(registry.formulas)
    custom = copy(registry.custom)
    for (name, implementation) in pairs
        haskey(formulas, name) && throw(ArgumentError("duplicate formula $name"))
        formulas[name] = implementation
        push!(custom, name)
    end
    return FormulaRegistry(formulas, custom)
end

"""A validated collection of method and theorem declarations."""
struct Catalogue
    methods::Vector{MethodDeclaration}
    theorems::Vector{TheoremDeclaration}
    formulas::FormulaRegistry
    warnings::Vector{CatalogueIssue}
end

const _KNOWN_OUTPUTS = Set((:last, :average, :best))
const _KNOWN_TIGHTNESS = Set((:tight, :known_upper_bound))
const _KNOWN_QUANTITIES = Set((
    :objective_gap,
    :distance_squared,
    :gradient_norm_squared,
    :gradient_mapping_norm_squared,
    :frank_wolfe_gap,
    :primal_dual_gap,
))
const _KNOWN_INITIAL_BOUNDS = union(
    _KNOWN_QUANTITIES,
    Set((
        :none,
        :fixed_point_distance_squared,
        :objective_gap_plus_half_mu_distance_squared,
        :primal_dual_distance_squared,
    )),
)
const _CATALOGUE_PROPERTY_PARAMETERS = Dict(
    :quadratic => (:lambda_min, :lambda_max),
    :centered_quadratic => (:coefficient,),
    :linear => (:operator_norm, :min_singular_value, :frame_constant),
    :linear_composition => (:operator_norm,),
)
const _KNOWN_PROPERTY_NAMES = union(
    Set(keys(_PROPERTY_PARAMETER)),
    Set(keys(_CATALOGUE_PROPERTY_PARAMETERS)),
    Set((:convex, :monotone)),
)
const _METHOD_FIELDS = Set((
    "id", "name", "parameter_vocabulary", "required_parameters", "calls"
))
const _CALL_FIELDS = Set(("role", "oracle", "count"))
const _THEOREM_FIELDS = Set((
    "id",
    "method",
    "name",
    "roles",
    "quantity",
    "output",
    "initial_bound",
    "bound",
    "parameter_rules",
    "parameter_domains",
    "citation",
    "tightness",
    "notes",
))
const _PARAMETER_DOMAIN_FIELDS = Set((
    "lower", "upper", "lower_closed", "upper_closed", "scale"
))
const _ROLE_FIELDS = Set(("name", "properties", "oracles"))
const _ORACLE_REQUIREMENT_FIELDS = Set(("name", "exactness", "inexactness"))
const _INEXACTNESS_FIELDS = Set((
    "model",
    "accuracy",
    "kappa",
    "inner_quantity",
    "inner_initial_bound",
    "inner_initial_value",
    "max_inner_iterations",
))
const _KNOWN_INEXACTNESS_MODELS = Set((:absolute_subproblem_gap,))

function _issue!(
    issues::Vector{CatalogueIssue},
    source::SourceLocation,
    field::AbstractString,
    message::AbstractString;
    severity::Symbol=:error,
)
    push!(issues, CatalogueIssue(severity, source, String(field), String(message)))
    return nothing
end

function _unknown_fields!(issues, source, table, allowed, prefix)
    for field in setdiff(Set(keys(table)), allowed)
        _issue!(issues, source, "$prefix.$field", "unknown field")
    end
end

function _required_string(issues, source, table, field; prefix="")
    path = isempty(prefix) ? field : "$prefix.$field"
    if !haskey(table, field)
        _issue!(issues, source, path, "required field is missing")
        return ""
    end
    value = table[field]
    if !(value isa AbstractString)
        _issue!(issues, source, path, "must be a string")
        return ""
    end
    isempty(value) && _issue!(issues, source, path, "must not be empty")
    return String(value)
end

function _string_vector(issues, source, table, field; prefix="")
    path = isempty(prefix) ? field : "$prefix.$field"
    if !haskey(table, field)
        _issue!(issues, source, path, "required field is missing")
        return String[]
    end
    value = table[field]
    if !(value isa AbstractVector)
        _issue!(issues, source, path, "must be an array of strings")
        return String[]
    end
    result = String[]
    for (index, item) in enumerate(value)
        if item isa AbstractString
            push!(result, String(item))
        else
            _issue!(issues, source, "$path[$index]", "must be a string")
        end
    end
    return result
end

function _table(issues, source, table, field; prefix="")
    path = isempty(prefix) ? field : "$prefix.$field"
    if !haskey(table, field)
        _issue!(issues, source, path, "required field is missing")
        return Dict{String,Any}()
    end
    value = table[field]
    if !(value isa AbstractDict)
        _issue!(issues, source, path, "must be a table")
        return Dict{String,Any}()
    end
    return Dict{String,Any}(String(key) => item for (key, item) in value)
end

function _tables(issues, source, table, field; prefix="")
    path = isempty(prefix) ? field : "$prefix.$field"
    if !haskey(table, field)
        _issue!(issues, source, path, "required field is missing")
        return Dict{String,Any}[]
    end
    value = table[field]
    if !(value isa AbstractVector)
        _issue!(issues, source, path, "must be an array of tables")
        return Dict{String,Any}[]
    end
    result = Dict{String,Any}[]
    for (index, item) in enumerate(value)
        if item isa AbstractDict
            push!(result, Dict{String,Any}(String(key) => entry for (key, entry) in item))
        else
            _issue!(issues, source, "$path[$index]", "must be a table")
        end
    end
    return result
end

function _parse_method(table, source, issues)
    _unknown_fields!(issues, source, table, _METHOD_FIELDS, "method")
    id = Symbol(_required_string(issues, source, table, "id"; prefix="method"))
    name = _required_string(issues, source, table, "name"; prefix="method")
    parameter_vocabulary = Symbol.(
        _string_vector(issues, source, table, "parameter_vocabulary"; prefix="method")
    )
    length(unique(parameter_vocabulary)) == length(parameter_vocabulary) ||
        _issue!(issues, source, "method.parameter_vocabulary", "contains duplicate names")
    required_parameters = Symbol.(
        _string_vector(issues, source, table, "required_parameters"; prefix="method")
    )
    length(unique(required_parameters)) == length(required_parameters) ||
        _issue!(issues, source, "method.required_parameters", "contains duplicate names")
    for parameter in setdiff(Set(required_parameters), Set(parameter_vocabulary))
        _issue!(
            issues,
            source,
            "method.required_parameters.$parameter",
            "is not in the method parameter vocabulary",
        )
    end
    call_tables = _tables(issues, source, table, "calls"; prefix="method")
    calls = OracleCall[]
    for (index, call_table) in enumerate(call_tables)
        path = "method.calls[$index]"
        _unknown_fields!(issues, source, call_table, _CALL_FIELDS, path)
        role = Symbol(_required_string(issues, source, call_table, "role"; prefix=path))
        oracle_name = Symbol(
            _required_string(issues, source, call_table, "oracle"; prefix=path)
        )
        count = get(call_table, "count", nothing)
        if count isa Integer && count > 0
            push!(calls, OracleCall(role, oracle_name, Int(count)))
        else
            _issue!(issues, source, "$path.count", "must be a positive integer")
        end
    end
    isempty(calls) && _issue!(issues, source, "method.calls", "must not be empty")
    call_keys = [(call.role, call.oracle) for call in calls]
    length(unique(call_keys)) == length(call_keys) ||
        _issue!(issues, source, "method.calls", "contains duplicate role/oracle pairs")
    return MethodDeclaration(
        id, name, parameter_vocabulary, required_parameters, calls, source
    )
end

function _parse_inexactness(table, source, issues, path)
    _unknown_fields!(issues, source, table, _INEXACTNESS_FIELDS, path)
    model = Symbol(_required_string(issues, source, table, "model"; prefix=path))
    model in _KNOWN_INEXACTNESS_MODELS ||
        _issue!(issues, source, "$path.model", "unknown inexactness model $model")
    accuracy = Symbol(_required_string(issues, source, table, "accuracy"; prefix=path))
    kappa = Symbol(_required_string(issues, source, table, "kappa"; prefix=path))
    inner_quantity = Symbol(
        _required_string(issues, source, table, "inner_quantity"; prefix=path)
    )
    inner_quantity in _KNOWN_QUANTITIES || _issue!(
        issues,
        source,
        "$path.inner_quantity",
        "unknown guarantee quantity $inner_quantity",
    )
    inner_initial_bound = Symbol(
        _required_string(issues, source, table, "inner_initial_bound"; prefix=path)
    )
    inner_initial_bound in _KNOWN_INITIAL_BOUNDS || _issue!(
        issues,
        source,
        "$path.inner_initial_bound",
        "unknown initial-bound quantity $inner_initial_bound",
    )
    raw_initial = get(table, "inner_initial_value", nothing)
    initial_value = if raw_initial isa Real
        scalar(raw_initial)
    elseif raw_initial isa AbstractString && !isempty(raw_initial)
        symbolic(Symbol(raw_initial); scope="$(source.file):$(source.entry)")
    else
        _issue!(
            issues,
            source,
            "$path.inner_initial_value",
            "must be a real number or a symbolic name",
        )
        scalar(1)
    end
    raw_limit = get(table, "max_inner_iterations", nothing)
    max_iterations = if raw_limit isa Integer && raw_limit > 0
        Int(raw_limit)
    else
        _issue!(issues, source, "$path.max_inner_iterations", "must be a positive integer")
        1
    end
    return ControlledInexactness(
        model,
        accuracy,
        kappa,
        InnerSolveRequirement(
            inner_quantity, inner_initial_bound, initial_value, max_iterations
        ),
    )
end

function _parse_oracle_requirement(table, source, issues, path)
    _unknown_fields!(issues, source, table, _ORACLE_REQUIREMENT_FIELDS, path)
    name = Symbol(_required_string(issues, source, table, "name"; prefix=path))
    exactness_name = Symbol(
        _required_string(issues, source, table, "exactness"; prefix=path)
    )
    exactness = get(_ORACLE_EXACTNESS_BY_NAME, exactness_name, nothing)
    if exactness === nothing
        _issue!(issues, source, "$path.exactness", "unknown exactness $exactness_name")
        exactness = ExactOracle
    end
    inexactness = if haskey(table, "inexactness")
        value = table["inexactness"]
        if value isa AbstractDict
            _parse_inexactness(
                Dict{String,Any}(String(key) => item for (key, item) in value),
                source,
                issues,
                "$path.inexactness",
            )
        else
            _issue!(issues, source, "$path.inexactness", "must be a table")
            nothing
        end
    else
        nothing
    end
    if exactness === ControlledInexactOracle
        if inexactness === nothing
            _issue!(
                issues,
                source,
                "$path.inexactness",
                "is required for a controlled-inexact oracle",
            )
            exactness = InexactOracle
        end
    elseif inexactness !== nothing
        _issue!(
            issues,
            source,
            "$path.inexactness",
            "is only valid for a controlled-inexact oracle",
        )
        inexactness = nothing
    end
    return OracleRequirement(name, exactness, inexactness)
end

function _parse_role(table, source, issues, index)
    path = "theorem.roles[$index]"
    _unknown_fields!(issues, source, table, _ROLE_FIELDS, path)
    name = Symbol(_required_string(issues, source, table, "name"; prefix=path))
    properties = Symbol.(_string_vector(issues, source, table, "properties"; prefix=path))
    length(unique(properties)) == length(properties) ||
        _issue!(issues, source, "$path.properties", "contains duplicate names")
    oracle_tables = _tables(issues, source, table, "oracles"; prefix=path)
    oracles = OracleRequirement[
        _parse_oracle_requirement(value, source, issues, "$path.oracles[$oracle_index]") for
        (oracle_index, value) in enumerate(oracle_tables)
    ]
    oracle_names = [value.name for value in oracles]
    length(unique(oracle_names)) == length(oracle_names) ||
        _issue!(issues, source, "$path.oracles", "contains duplicate names")
    return RoleRequirement(name, properties, oracles)
end

function _parameter_endpoint(issues, source, table, field, path, fallback)
    if !haskey(table, field)
        _issue!(issues, source, "$path.$field", "required field is missing")
        return fallback
    end
    value = table[field]
    if value isa Real
        isfinite(value) || _issue!(issues, source, "$path.$field", "must be finite")
        return Float64(value)
    elseif value isa AbstractString && !isempty(value)
        return Symbol(value)
    end
    _issue!(issues, source, "$path.$field", "must be a real number or formula name")
    return fallback
end

function _parameter_domain_flag(issues, source, table, field, path)
    value = get(table, field, true)
    if !(value isa Bool)
        _issue!(issues, source, "$path.$field", "must be a boolean")
        return true
    end
    return value
end

function _parse_parameter_domain(table, source, issues, parameter)
    path = "theorem.parameter_domains.$parameter"
    _unknown_fields!(issues, source, table, _PARAMETER_DOMAIN_FIELDS, path)
    lower = _parameter_endpoint(issues, source, table, "lower", path, 0.0)
    upper = _parameter_endpoint(issues, source, table, "upper", path, 1.0)
    lower_closed = _parameter_domain_flag(
        issues, source, table, "lower_closed", path
    )
    upper_closed = _parameter_domain_flag(
        issues, source, table, "upper_closed", path
    )
    raw_scale = get(table, "scale", "linear")
    scale = if raw_scale isa AbstractString
        Symbol(raw_scale)
    else
        _issue!(issues, source, "$path.scale", "must be a string")
        :linear
    end
    if !(scale in (:linear, :log))
        _issue!(issues, source, "$path.scale", "must be linear or log")
        scale = :linear
    end
    try
        return ParameterDomain(
            lower,
            upper;
            lower_closed=lower_closed,
            upper_closed=upper_closed,
            scale=scale,
        )
    catch error
        _issue!(issues, source, path, sprint(showerror, error))
        return ParameterDomain(0.0, 1.0)
    end
end

function _parse_theorem(table, source, issues)
    _unknown_fields!(issues, source, table, _THEOREM_FIELDS, "theorem")
    id = Symbol(_required_string(issues, source, table, "id"; prefix="theorem"))
    method = Symbol(_required_string(issues, source, table, "method"; prefix="theorem"))
    name = _required_string(issues, source, table, "name"; prefix="theorem")
    role_tables = _tables(issues, source, table, "roles"; prefix="theorem")
    roles = RoleRequirement[
        _parse_role(value, source, issues, index) for
        (index, value) in enumerate(role_tables)
    ]
    isempty(roles) && _issue!(issues, source, "theorem.roles", "must not be empty")
    role_names = [value.name for value in roles]
    length(unique(role_names)) == length(role_names) ||
        _issue!(issues, source, "theorem.roles", "contains duplicate role names")
    quantity = Symbol(_required_string(issues, source, table, "quantity"; prefix="theorem"))
    quantity in _KNOWN_QUANTITIES ||
        _issue!(issues, source, "theorem.quantity", "unknown guarantee quantity $quantity")
    output = Symbol(_required_string(issues, source, table, "output"; prefix="theorem"))
    output in _KNOWN_OUTPUTS ||
        _issue!(issues, source, "theorem.output", "unknown output convention $output")
    initial_bound = Symbol(
        _required_string(issues, source, table, "initial_bound"; prefix="theorem")
    )
    initial_bound in _KNOWN_INITIAL_BOUNDS || _issue!(
        issues,
        source,
        "theorem.initial_bound",
        "unknown initial-bound quantity $initial_bound",
    )
    bound = Symbol(_required_string(issues, source, table, "bound"; prefix="theorem"))
    raw_rules = _table(issues, source, table, "parameter_rules"; prefix="theorem")
    parameter_rules = Dict{Symbol,Symbol}()
    for (parameter, formula) in raw_rules
        if formula isa AbstractString
            parameter_rules[Symbol(parameter)] = Symbol(formula)
        else
            _issue!(
                issues, source, "theorem.parameter_rules.$parameter", "must name a formula"
            )
        end
    end
    raw_domains = if haskey(table, "parameter_domains")
        value = table["parameter_domains"]
        if value isa AbstractDict
            Dict{String,Any}(String(key) => item for (key, item) in value)
        else
            _issue!(issues, source, "theorem.parameter_domains", "must be a table")
            Dict{String,Any}()
        end
    else
        Dict{String,Any}()
    end
    parameter_domains = Dict{Symbol,ParameterDomain}()
    for (parameter, raw_domain) in raw_domains
        if raw_domain isa AbstractDict
            domain_table = Dict{String,Any}(
                String(key) => item for (key, item) in raw_domain
            )
            parameter_domains[Symbol(parameter)] = _parse_parameter_domain(
                domain_table, source, issues, parameter
            )
        else
            _issue!(
                issues,
                source,
                "theorem.parameter_domains.$parameter",
                "must be a table",
            )
        end
    end
    for parameter in intersect(Set(keys(parameter_rules)), Set(keys(parameter_domains)))
        _issue!(
            issues,
            source,
            "theorem.parameter_domains.$parameter",
            "cannot also have a fixed parameter rule",
        )
    end
    citation = _required_string(issues, source, table, "citation"; prefix="theorem")
    tightness = Symbol(
        _required_string(issues, source, table, "tightness"; prefix="theorem")
    )
    tightness in _KNOWN_TIGHTNESS ||
        _issue!(issues, source, "theorem.tightness", "unknown status $tightness")
    notes = if haskey(table, "notes")
        value = table["notes"]
        if value isa AbstractDict
            Dict{String,Any}(String(key) => item for (key, item) in value)
        else
            _issue!(issues, source, "theorem.notes", "must be a table")
            Dict{String,Any}()
        end
    else
        Dict{String,Any}()
    end
    return TheoremDeclaration(
        id,
        method,
        name,
        roles,
        quantity,
        output,
        initial_bound,
        bound,
        parameter_rules,
        parameter_domains,
        citation,
        tightness,
        notes,
        source,
    )
end

function _parse_entries(path::AbstractString, top_level::String, issues, parser)
    document = try
        TOML.parsefile(path)
    catch error
        source = SourceLocation(String(path), 0)
        _issue!(issues, source, top_level, sprint(showerror, error))
        return Any[]
    end
    for field in setdiff(Set(keys(document)), Set((top_level,)))
        _issue!(issues, SourceLocation(String(path), 0), field, "unknown top-level field")
    end
    raw_entries = get(document, top_level, nothing)
    if !(raw_entries isa AbstractVector)
        _issue!(
            issues, SourceLocation(String(path), 0), top_level, "must be an array of tables"
        )
        return Any[]
    end
    entries = Any[]
    for (index, raw) in enumerate(raw_entries)
        source = SourceLocation(String(path), index)
        if raw isa AbstractDict
            table = Dict{String,Any}(String(key) => value for (key, value) in raw)
            push!(entries, parser(table, source, issues))
        else
            _issue!(issues, source, top_level, "entry must be a table")
        end
    end
    return entries
end

function _resolve_formulas(theorems, additions::FormulaRegistry, issues)
    names = Set{Symbol}(theorem.bound for theorem in theorems)
    for theorem in theorems
        union!(names, values(theorem.parameter_rules))
        for domain in values(theorem.parameter_domains)
            domain.lower isa Symbol && push!(names, domain.lower)
            domain.upper isa Symbol && push!(names, domain.upper)
        end
    end
    formulas = copy(additions.formulas)
    for name in names
        haskey(formulas, name) && continue
        if isdefined(@__MODULE__, name)
            implementation = getfield(@__MODULE__, name)
            if implementation isa Function
                formulas[name] = implementation
                continue
            end
        end
        theorem = findfirst(
            value ->
                value.bound === name ||
                    name in values(value.parameter_rules) ||
                    any(
                        domain -> domain.lower === name || domain.upper === name,
                        values(value.parameter_domains),
                    ),
            theorems,
        )
        source = if theorem === nothing
            SourceLocation("<catalogue>", 0)
        else
            theorems[theorem].source
        end
        _issue!(issues, source, "formula.$name", "no Julia implementation is registered")
    end
    return FormulaRegistry(formulas, copy(additions.custom))
end

function _declared_property_parameters(property_name::Symbol)
    parameters = Symbol[]
    primary = get(_PROPERTY_PARAMETER, property_name, nothing)
    primary === nothing || push!(parameters, primary)
    append!(parameters, get(_CATALOGUE_PROPERTY_PARAMETERS, property_name, ()))
    return unique(parameters)
end

function _qualified_formula_input_name(source::FormulaInputSource)
    Symbol(source.role, "__", source.property, "__", source.parameter)
end

function _parse_qualified_formula_input_name(name::Symbol)
    parts = split(string(name), "__"; limit=3)
    length(parts) == 3 || return nothing
    return FormulaInputSource(Symbol.(parts)...)
end

function _formula_input_aliases(sources::Vector{FormulaInputSource})
    aliases = Pair{Symbol,Int}[]
    by_role = Dict{Tuple{Symbol,Symbol},Vector{Int}}()
    by_name = Dict{Symbol,Vector{Int}}()
    for (index, source) in enumerate(sources)
        push!(aliases, _qualified_formula_input_name(source) => index)
        push!(get!(by_role, (source.role, source.parameter), Int[]), index)
        push!(get!(by_name, source.parameter, Int[]), index)
    end
    for ((role, parameter), indices) in by_role
        length(indices) == 1 &&
            push!(aliases, Symbol(role, "_", parameter) => only(indices))
    end
    for (parameter, indices) in by_name
        length(indices) == 1 && push!(aliases, parameter => only(indices))
    end
    return aliases
end

function _available_formula_inputs(theorem::TheoremDeclaration)
    sources = FormulaInputSource[]
    for role in theorem.roles, property_name in role.properties
        for parameter in _declared_property_parameters(property_name)
            push!(sources, FormulaInputSource(role.name, property_name, parameter))
        end
    end
    available = Set((:k, :initial))
    union!(available, keys(theorem.parameter_rules))
    union!(available, keys(theorem.parameter_domains))
    for (alias, _) in _formula_input_aliases(sources)
        push!(available, alias)
    end
    for role in theorem.roles, requirement in role.oracles
        requirement.inexactness === nothing && continue
        push!(available, requirement.inexactness.accuracy)
        push!(available, requirement.inexactness.kappa)
    end
    return available
end

function _parameter_domain_formula_inputs(
    theorem::TheoremDeclaration, registry::FormulaRegistry, cache
)
    result = Dict{Tuple{Symbol,Symbol},Set{Symbol}}()
    for (parameter, domain) in theorem.parameter_domains
        for (endpoint, formula_name) in ((:lower, domain.lower), (:upper, domain.upper))
            formula_name isa Symbol || continue
            implementation = get(registry.formulas, formula_name, nothing)
            result[(parameter, endpoint)] = if implementation === nothing
                Set{Symbol}()
            else
                get!(cache, formula_name) do
                    _required_formula_inputs(implementation)
                end
            end
        end
    end
    return result
end

function _qualified_formula_input(theorem::TheoremDeclaration, input::Symbol)
    source = _parse_qualified_formula_input_name(input)
    source === nothing && return false
    role = findfirst(requirement -> requirement.name === source.role, theorem.roles)
    role === nothing && return false
    source.property in theorem.roles[role].properties || return false
    source.property in _KNOWN_PROPERTY_NAMES || return true
    return source.parameter in _declared_property_parameters(source.property)
end

_context_input(name::Symbol) = name === Symbol("γ") ? :gamma : name

function _formula_keyword_contract(implementation::Function)
    keywords = Dict{Symbol,Symbol}()
    accepts_extra = false
    for method in methods(implementation)
        method.nargs == 1 || continue
        for name in Base.kwarg_decl(method)
            if endswith(string(name), "...")
                accepts_extra = true
                continue
            end
            keywords[_context_input(name)] = name
        end
    end
    return keywords, accepts_extra
end

function _required_formula_inputs(implementation::Function)
    return Set(keys(first(_formula_keyword_contract(implementation))))
end

function _parameter_formula_inputs(parameter_rules, registry::FormulaRegistry, cache)
    result = Dict{Symbol,Set{Symbol}}()
    for (parameter, formula_name) in parameter_rules
        implementation = get(registry.formulas, formula_name, nothing)
        result[parameter] = if implementation === nothing
            Set{Symbol}()
        else
            get!(cache, formula_name) do
                _required_formula_inputs(implementation)
            end
        end
    end
    return result
end

function _parameter_evaluation_order(inputs::Dict{Symbol,Set{Symbol}})
    parameters = Set(keys(inputs))
    dependencies = Dict(
        parameter => intersect(required, parameters) for (parameter, required) in inputs
    )
    order = Symbol[]
    resolved = Set{Symbol}()
    remaining = copy(parameters)
    while !isempty(remaining)
        ready = sort!(
            Symbol[
                parameter for
                parameter in remaining if issubset(dependencies[parameter], resolved)
            ];
            by=string,
        )
        isempty(ready) && break
        append!(order, ready)
        union!(resolved, ready)
        setdiff!(remaining, ready)
    end
    return order, remaining
end

function _validate_bound_formula_inputs!(
    theorems, registry::FormulaRegistry, additions::FormulaRegistry, issues
)
    required = Dict{Symbol,Set{Symbol}}()
    custom = additions.custom
    for theorem in theorems
        formula_name = theorem.bound
        formula_name in custom || continue
        implementation = get(registry.formulas, formula_name, nothing)
        implementation === nothing && continue
        available = _available_formula_inputs(theorem)
        inputs = get!(required, formula_name) do
            _required_formula_inputs(implementation)
        end
        for input in setdiff(inputs, available)
            _qualified_formula_input(theorem, input) && continue
            _issue!(
                issues,
                theorem.source,
                "theorem.bound",
                "formula $formula_name requires undeclared input $input",
            )
        end
    end
    return nothing
end

function _validate_parameter_dependencies!(theorems, registry::FormulaRegistry, issues)
    required = Dict{Symbol,Set{Symbol}}()
    for theorem in theorems
        parameters = union(
            Set(keys(theorem.parameter_rules)), Set(keys(theorem.parameter_domains))
        )
        external = setdiff(_available_formula_inputs(theorem), parameters)
        inputs = _parameter_formula_inputs(theorem.parameter_rules, registry, required)
        for (parameter, formula_inputs) in inputs
            formula_name = theorem.parameter_rules[parameter]
            for input in setdiff(formula_inputs, union(external, parameters))
                _qualified_formula_input(theorem, input) && continue
                _issue!(
                    issues,
                    theorem.source,
                    "theorem.parameter_rules.$parameter",
                    "formula $formula_name requires undeclared input $input",
                )
            end
        end
        _, remaining = _parameter_evaluation_order(inputs)
        isempty(remaining) || _issue!(
            issues,
            theorem.source,
            "theorem.parameter_rules",
            "contains circular parameter dependencies: $(join(sort!(string.(collect(remaining))), ", "))",
        )
        domain_inputs = _parameter_domain_formula_inputs(theorem, registry, required)
        for ((parameter, endpoint), formula_inputs) in domain_inputs
            domain = theorem.parameter_domains[parameter]
            formula_name = getfield(domain, endpoint)
            for input in setdiff(formula_inputs, external)
                _qualified_formula_input(theorem, input) && continue
                _issue!(
                    issues,
                    theorem.source,
                    "theorem.parameter_domains.$parameter.$endpoint",
                    "formula $formula_name requires undeclared or selected-parameter input $input",
                )
            end
        end
    end
    return nothing
end

function _cross_validate!(methods, theorems, issues)
    method_ids = Dict{Symbol,MethodDeclaration}()
    for method in methods
        if haskey(method_ids, method.id)
            _issue!(issues, method.source, "method.id", "duplicate id $(method.id)")
        else
            method_ids[method.id] = method
        end
    end
    theorem_ids = Set{Symbol}()
    for theorem in theorems
        if theorem.id in theorem_ids
            _issue!(issues, theorem.source, "theorem.id", "duplicate id $(theorem.id)")
        else
            push!(theorem_ids, theorem.id)
        end
        method = get(method_ids, theorem.method, nothing)
        if method === nothing
            _issue!(
                issues, theorem.source, "theorem.method", "unknown method $(theorem.method)"
            )
            continue
        end
        selected_parameters = union(
            Set(keys(theorem.parameter_rules)), Set(keys(theorem.parameter_domains))
        )
        unknown_parameters = setdiff(selected_parameters, Set(method.parameter_vocabulary))
        for parameter in unknown_parameters
            _issue!(
                issues,
                theorem.source,
                "theorem.parameters.$parameter",
                "is not declared by method $(method.id)",
            )
        end
        missing_parameters = setdiff(
            Set(method.required_parameters), selected_parameters
        )
        for parameter in missing_parameters
            _issue!(
                issues,
                theorem.source,
                "theorem.parameters.$parameter",
                "is required by method $(method.id)",
            )
        end
        role_ids = Set(role.name for role in theorem.roles)
        for call in method.calls
            if !(call.role in role_ids)
                _issue!(
                    issues,
                    theorem.source,
                    "theorem.roles",
                    "method $(method.id) calls undeclared role $(call.role)",
                )
                continue
            end
            role = only(filter(value -> value.name === call.role, theorem.roles))
            call.oracle in (requirement.name for requirement in role.oracles) || _issue!(
                issues,
                theorem.source,
                "theorem.roles.$(call.role).oracles",
                "method $(method.id) calls undeclared oracle $(call.oracle)",
            )
        end
    end
    for method in methods
        method_theorems = filter(theorem -> theorem.method === method.id, theorems)
        isempty(method_theorems) && _issue!(
            issues,
            method.source,
            "method.id",
            "method $(method.id) is not referenced by any theorem",
        )
        used_parameters = Set{Symbol}()
        for theorem in method_theorems
            union!(used_parameters, keys(theorem.parameter_rules))
            union!(used_parameters, keys(theorem.parameter_domains))
        end
        unused_parameters = setdiff(Set(method.parameter_vocabulary), used_parameters)
        for parameter in unused_parameters
            _issue!(
                issues,
                method.source,
                "method.parameter_vocabulary.$parameter",
                "is not selected by any theorem for method $(method.id)",
            )
        end
    end
    return nothing
end

"""
    load_catalogue(root; formulas=FormulaRegistry())

Load and strictly validate `methods.toml` and every file in `theorems/`.
"""
function load_catalogue(root::AbstractString; formulas::FormulaRegistry=FormulaRegistry())
    issues = CatalogueIssue[]
    method_path = joinpath(root, "methods.toml")
    isfile(method_path) && Base.include_dependency(method_path)
    isfile(method_path) ||
        _issue!(issues, SourceLocation(method_path, 0), "methods", "file does not exist")
    methods = if isfile(method_path)
        MethodDeclaration[_parse_entries(method_path, "methods", issues, _parse_method)...]
    else
        MethodDeclaration[]
    end
    theorem_directory = joinpath(root, "theorems")
    isdir(theorem_directory) && Base.include_dependency(theorem_directory)
    theorem_paths = if isdir(theorem_directory)
        sort!(filter(path -> endswith(path, ".toml"), readdir(theorem_directory; join=true)))
    else
        String[]
    end
    isempty(theorem_paths) && _issue!(
        issues,
        SourceLocation(theorem_directory, 0),
        "theorems",
        "no theorem TOML files were found",
    )
    theorems = TheoremDeclaration[]
    for path in theorem_paths
        Base.include_dependency(path)
        append!(theorems, _parse_entries(path, "theorems", issues, _parse_theorem))
    end
    _cross_validate!(methods, theorems, issues)
    registry = _resolve_formulas(theorems, formulas, issues)
    _validate_bound_formula_inputs!(theorems, registry, formulas, issues)
    _validate_parameter_dependencies!(theorems, registry, issues)
    errors = CatalogueIssue[issue for issue in issues if issue.severity === :error]
    isempty(errors) || throw(InvalidCatalogue(errors))
    warnings = CatalogueIssue[issue for issue in issues if issue.severity === :warning]
    return Catalogue(methods, theorems, registry, warnings)
end

function _replace_by_id(
    base::Vector{T}, addition::Vector{T}, replace::Bool, label
) where {T}
    result = copy(base)
    positions = Dict(getfield(value, :id) => index for (index, value) in enumerate(result))
    for value in addition
        identifier = getfield(value, :id)
        if haskey(positions, identifier)
            replace || throw(ArgumentError("duplicate $label id $identifier"))
            result[positions[identifier]] = value
        else
            push!(result, value)
            positions[identifier] = length(result)
        end
    end
    return result
end

"""Compose validated catalogues; replacing declarations requires `replace=true`."""
function compose_catalogues(base::Catalogue, additions::Catalogue...; replace::Bool=false)
    methods = copy(base.methods)
    theorems = copy(base.theorems)
    formulas = copy(base.formulas.formulas)
    custom_formulas = copy(base.formulas.custom)
    warnings = copy(base.warnings)
    for addition in additions
        methods = _replace_by_id(methods, addition.methods, replace, "method")
        theorems = _replace_by_id(theorems, addition.theorems, replace, "theorem")
        for (name, implementation) in addition.formulas.formulas
            if haskey(formulas, name) && formulas[name] !== implementation
                replace || throw(ArgumentError("duplicate formula $name"))
            end
            formulas[name] = implementation
        end
        union!(custom_formulas, addition.formulas.custom)
        append!(warnings, addition.warnings)
    end
    issues = CatalogueIssue[]
    _cross_validate!(methods, theorems, issues)
    registry = FormulaRegistry(formulas, custom_formulas)
    _validate_bound_formula_inputs!(theorems, registry, registry, issues)
    _validate_parameter_dependencies!(theorems, registry, issues)
    isempty(issues) || throw(InvalidCatalogue(issues))
    return Catalogue(methods, theorems, registry, warnings)
end

const DEFAULT_CATALOGUE_PATH = normpath(joinpath(@__DIR__, "..", "catalogue"))
const DEFAULT_CATALOGUE = load_catalogue(DEFAULT_CATALOGUE_PATH)

"""Return an independent copy of the validated first-party catalogue."""
default_catalogue() = deepcopy(DEFAULT_CATALOGUE)

function method_by_id(catalogue::Catalogue, id::Symbol)
    (index -> index === nothing ? nothing : catalogue.methods[index])(
        findfirst(method -> method.id === id, catalogue.methods)
    )
end

function theorem_by_id(catalogue::Catalogue, id::Symbol)
    (index -> index === nothing ? nothing : catalogue.theorems[index])(
        findfirst(theorem -> theorem.id === id, catalogue.theorems)
    )
end

formula(registry::FormulaRegistry, name::Symbol) = get(registry.formulas, name) do
    throw(KeyError(name))
end
