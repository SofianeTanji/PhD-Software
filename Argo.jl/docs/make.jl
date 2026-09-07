using Documenter

const ROOT = normpath(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, ROOT)

using Argo

makedocs(;
    sitename="Argo.jl",
    modules=[Argo],
    checkdocs=:none,
    format=Documenter.HTML(; prettyurls=true, edit_link=nothing),
    pages=[
        "Guide" => "index.md",
        "Modeling" => "modeling.md",
        "Catalogue and certificates" => "catalogue.md",
        "Capabilities" => "capabilities.md",
        "Authoring" => "authoring.md",
    ],
)
