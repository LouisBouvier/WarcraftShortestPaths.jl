using WarcraftShortestPaths
using CairoMakie
using Images
# using Graphs
# using GridGraphs
# using Flux
# using Images
# using LinearAlgebra
# using Plots
# using Random
# using Statistics
# using Test

Random.seed!(63);

path = joinpath(@__DIR__, "..", "data", "warcraft_maps.tar.gz")
decompressed_path = joinpath(@__DIR__, "..", "data", "warcraft_maps")
# WarcraftShortestPaths.decompress_dataset(path, decompressed_path)

options = (ϵ=0.1, M=3, nb_epochs=50, nb_samples=50, batch_size=7, lr_start=0.001)

dataset = create_dataset(decompressed_path, options.nb_samples);
train_dataset, test_dataset = train_test_split(dataset, 0.8);

x, y_true, kwargs = test_dataset[6];

map_img = colorview(RGB, permutedims(dropdims(x; dims=4), (3, 1, 2)))
weights = kwargs.wg.weights
path = Gray.(0.7 .* y_true)

# Plots Guillaume

f, ax, pl = image(map_img'; axis=(aspect=DataAspect(), yreversed=true))
colsize!(f.layout, 1, Aspect(1, 1.0))
resize_to_layout!(f)
hidedecorations!(ax)
hidespines!(ax)
f
save("map.png", f)

f, ax, pl = heatmap(weights'; colormap=:thermal, axis=(aspect=DataAspect(), yreversed=true))
Colorbar(f[:, 2], pl)
colsize!(f.layout, 1, Aspect(1, 1.0))
resize_to_layout!(f)
hidedecorations!(ax)
hidespines!(ax)
f
save("weights.png", f)

f, ax, pl = image(path'; interpolate=false, axis=(aspect=DataAspect(), yreversed=true))
colsize!(f.layout, 1, Aspect(1, 1.0))
resize_to_layout!(f)
hidedecorations!(ax)
hidespines!(ax)
f
save("path.png", f)
