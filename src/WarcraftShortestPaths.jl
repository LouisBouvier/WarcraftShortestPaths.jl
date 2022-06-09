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
export train_with_perturbed_FYL!
export train_with_perturbed_cost!
export plot_loss
export linear_maximizer
export convert_image_for_plot
export plot_image_and_path
export plot_image_label_path
export plot_terrain_weights
export plot_loss_and_cost_ratio

end
