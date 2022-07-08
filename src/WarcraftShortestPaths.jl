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
using Tar

include("warcraft_plots.jl")
include("embedding.jl")
include("dataset.jl")
include("train.jl")

export create_dataset
export train_test_split

export create_warcraft_embedding
export true_maximizer
export cost

export train_function!

export convert_image_for_plot
export plot_loss_and_gap
export plot_map
export plot_path
export plot_weights
export plot_image_label_path
export plot_terrain_weights

end
