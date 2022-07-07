module WarcraftShortestPaths

using Colors
using Flux
using GZip
using Graphs
using GridGraphs
using Images
using InferOpt
using JSON
using LinearAlgebra
using Metalhead
using NPZ
using Plots
using ProgressMeter
using Random
using Statistics
using Tar
using UnicodePlots


include("warcraft_plots.jl")
include("embedding.jl")
include("dataset.jl")
include("error.jl")
include("perf.jl")
include("pipeline.jl")

export generate_dataset
export create_warcraft_embedding

export plot_map
export plot_weights
export plot_path

export apply_learning_pipeline!

end
