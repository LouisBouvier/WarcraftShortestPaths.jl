using WarcraftShortestPaths
using Documenter

DocMeta.setdocmeta!(WarcraftShortestPaths, :DocTestSetup, :(using WarcraftShortestPaths); recursive=true)

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
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/LouisBouvier/WarcraftShortestPaths.jl",
    devbranch="main",
)
