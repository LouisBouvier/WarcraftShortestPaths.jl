using WarcraftShortestPaths
using Graphs
using GridGraphs
using Flux
using Images
using LinearAlgebra
using Plots
using Random
using Statistics
using Test

Random.seed!(63);

path = joinpath(@__DIR__, "..", "data", "warcraft_maps.tar.gz")
decompressed_path = joinpath(@__DIR__, "..", "data", "warcraft_maps")
# WarcraftShortestPaths.decompress_dataset(path, decompressed_path)

options = (ϵ=0.1, M=3, nb_epochs=50, nb_samples=50, batch_size=7, lr_start=0.001)

dataset = create_dataset(decompressed_path, options.nb_samples)
train_dataset, test_dataset = train_test_split(dataset, 0.8)

x, y_true, kwargs = test_dataset[6]
im = convert_image_for_plot(x[:, :, :, 1])
plot_image_and_path(im, y_true)

model = create_warcraft_embedding()

Losses, Cost_ratios = train_with_perturbed_FYL!(;
    model=model,
    train_dataset=Flux.DataLoader(train_dataset; batchsize=options.batch_size),
    test_dataset=Flux.DataLoader(test_dataset; batchsize=length(test_dataset)),
    options=options,
)

Gaps = Cost_ratios .- 1
plot_loss_and_cost_ratio(Losses, Gaps, options)

θ_test = model(x)
shortest_path = UInt8.(linear_maximizer(θ_test))

plot_image_label_path(im, shortest_path, y_true)

plot_terrain_weights(kwargs.wg.weights, -θ_test)

# Plots Guillaume

function plot_map_giom(map_matrix::Array{<:Real,3}; filepath=nothing)
    img = convert_image_for_plot(map_matrix)
    pl = Plots.plot(
        img;
        aspect_ratio=:equal,
        framestyle=:none,
        size=(500, 500)
    )
    isnothing(filepath) || Plots.savefig(pl, filepath)
    return pl
end

function plot_weights_giom(weights::Matrix{<:Real}; filepath=nothing)
    pl = Plots.heatmap(
        weights;
        yflip=true,
        aspect_ratio=:equal,
        framestyle=:none,
        padding=(0., 0.),
        size=(500, 500)
    )
    isnothing(filepath) || Plots.savefig(pl, filepath)
    return pl
end

function plot_path_giom(path::Matrix{<:Integer}; filepath=nothing)
    pl = Plots.plot(
        Gray.(path .* 0.7);
        aspect_ratio=:equal,
        framestyle=:none,
        size=(500, 500)
    )
    isnothing(filepath) || Plots.savefig(pl, filepath)
    return pl
end

plot_map_giom(dropdims(x; dims=4), filepath="map.pdf")
plot_weights_giom(kwargs.wg.weights, filepath="cost.pdf")
plot_path_giom(y_true, filepath="path.pdf")
