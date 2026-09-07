Base.@kwdef struct ThermalGen{
    TF<:AbstractFloat,
    TI<:Integer,
    AVF<:AbstractVector{TF},
    AVI<:AbstractVector{TI},
}
    MinRunCapacity::AVF
    MaxRunCapacity::AVF
    RampUp::AVF
    RampDown::AVF
    StartUp::AVF
    ShutDown::AVF
    UpTime::AVI
    DownTime::AVI
    NoLoadConsumption::AVF
    MarginalCost::AVF
    FixedCost::AVF
end

Base.@kwdef struct Instance{
    TF<:AbstractFloat,
    TI<:Integer,
    ALoad<:AbstractArray{TF},
    TG<:ThermalGen{TF,TI,<:AbstractVector{TF},<:AbstractVector{TI}},
}
    LostLoad::TF
    Load::ALoad
    ThermalGen::TG
end

# Note: We keep these parametric because JuMP variable/constraint container types
# differ depending on how you index (Vector, DenseAxisArray, SparseAxisArray, etc.).
struct SubProblem{M,VP,VU,VV,VW,VC,CL,CMU,CMD,GL1,GL2,RU,RD}
    model::M
    Varp::VP
    Varu::VU
    Varv::VV
    Varw::VW
    VarCost::VC
    ConstrLogical::CL
    ConstrMinUpTime::CMU
    ConstrMinDownTime::CMD
    ConstrGenLimits1::GL1
    ConstrGenLimits2::GL2
    ConstrRampUp::RU
    ConstrRampDown::RD
end

struct RestrictedMasterProgram{M,VZ,VL,CB,CC}
    model::M
    VarZ::VZ
    VarL::VL
    ConstrBalance::CB
    ConstrConvexComb::CC
end
