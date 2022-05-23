using WarcraftShortestPath
using Documenter

DocMeta.setdocmeta!(WarcraftShortestPath, :DocTestSetup, :(using WarcraftShortestPath); recursive=true)

makedocs(;
    modules=[WarcraftShortestPath],
    authors="LouisBouvier <lbouvier975@gmail.com> and contributors",
    repo="https://github.com/LouisBouvier/WarcraftShortestPath.jl/blob/{commit}{path}#{line}",
    sitename="WarcraftShortestPath.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://LouisBouvier.github.io/WarcraftShortestPath.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/LouisBouvier/WarcraftShortestPath.jl",
    devbranch="main",
)
