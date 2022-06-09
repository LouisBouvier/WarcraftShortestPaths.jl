using WarcraftShortestPaths
using Documenter
using Literate

DocMeta.setdocmeta!(WarcraftShortestPaths, :DocTestSetup, :(using WarcraftShortestPaths); recursive=true)

# Parse test/tutorial.jl into docs/src/tutorial.md (overwriting)

tuto_jl_file = joinpath(dirname(@__DIR__), "test", "tutorial.jl")
tuto_md_dir = joinpath(@__DIR__, "src")
Literate.markdown(tuto_jl_file, tuto_md_dir; documenter=true, execute=false)

makedocs(;
    modules=[WarcraftShortestPaths],
    authors="LouisBouvier <lbouvier975@gmail.com> and contributors",
    repo="https://github.com/LouisBouvier/WarcraftShortestPaths.jl/blob/{commit}{path}#{line}",
    sitename="WarcraftShortestPaths.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://LouisBouvier.github.io/WarcraftShortestPaths.jl",
        assets=String[],
    ),
    pages=[
        "Overview" => "index.md",
        "API reference" => "API.md",
        "Tutorial" => "tutorial.md",
    ],
)

deploydocs(;
    repo="github.com/LouisBouvier/WarcraftShortestPaths.jl",
    devbranch="main",
)
