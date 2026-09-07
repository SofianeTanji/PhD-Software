"""
    load_data(JSON_file; ramp_scale=1.0, load_scale=1.0, lost_load=3_000.0)

Load a unit-commitment instance from a JSON file and return an `Instance`.

# Keyword arguments
- `ramp_scale` : RampUp and RampDown values are divided by this (default 1.0).
- `load_scale` : The demand vector is divided by this (default 1.0).
- `lost_load`  : Value of lost load (default 3_000.0).
"""
function load_data(
    JSON_file;
    ramp_scale::Float64 = 1.0,
    load_scale::Float64 = 1.0,
    lost_load::Float64 = 3_000.0,
)
    df = DataFrame(JSON3.read(JSON_file))
    Gen = df.thermal_generators[1]

    MinRunCapacity = Float64[]
    MaxRunCapacity = Float64[]
    RampUp = Float64[]
    RampDown = Float64[]
    UpTime = Int[]
    DownTime = Int[]
    StartUp = Float64[]
    ShutDown = Float64[]
    NoLoadConsumption = Float64[]
    FixedCost = Float64[]
    MarginalCost = Float64[]

    for (_sym, gen) in Gen
        push!(MinRunCapacity, gen.power_output_minimum)
        push!(MaxRunCapacity, gen.power_output_maximum)
        push!(RampUp, gen.ramp_up_limit / ramp_scale)
        push!(RampDown, gen.ramp_down_limit / ramp_scale)
        push!(UpTime, gen.time_up_minimum)
        push!(DownTime, gen.time_down_minimum)

        push!(
            StartUp,
            haskey(gen, "ramp_startup_limit") ? gen.ramp_startup_limit :
            gen.power_output_minimum,
        )
        push!(
            ShutDown,
            haskey(gen, "ramp_shutdown_limit") ? gen.ramp_shutdown_limit :
            gen.power_output_minimum,
        )
        push!(
            NoLoadConsumption,
            haskey(gen, "no_load_consumption") ? gen.no_load_consumption :
            gen.time_down_minimum,
        )

        push!(FixedCost, gen.startup[1]["cost"])
        push!(MarginalCost, gen.piecewise_production[1]["cost"])
    end

    thermal = ThermalGen(;
        MinRunCapacity,
        MaxRunCapacity,
        RampUp,
        RampDown,
        StartUp,
        ShutDown,
        UpTime,
        DownTime,
        NoLoadConsumption,
        MarginalCost,
        FixedCost,
    )

    return Instance(;
        LostLoad = lost_load,
        Load = Array{Float64}(df.demand) ./ load_scale,
        ThermalGen = thermal,
    )
end

"""Load a California instance (LostLoad = 15 000)."""
load_ca_data(JSON_file) = load_data(JSON_file; lost_load = 15_000.0)
