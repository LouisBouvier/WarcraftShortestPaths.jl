# # Tutorial

# ## Context

#=
In this tutorial page we consider the learning by imitation setting. We have a sub-dataset of Warcraft terrain 
images and corresponding shortest paths. We want to learn the cost of the cells, using a neural network embedding.


More precisely, each point in our dataset consists in:
- an image of terrain ``I``
- a shortest path ``P`` from the top left to the bottom right corner

We don't know the true costs that were used to compute the shortest path, but we can exploit the images to approximate these costs.
The question is: how should we combine these features?

We will use `InferOpt` to learn the appropriate weights, so that we may propose relevant paths in the future.
=#

using WarcraftShortestPaths
using Graphs
using GridGraphs
using Flux
using LinearAlgebra
using Random
using Statistics
using Test
using UnicodePlots

Random.seed!(63);
decompressed_path = joinpath(@__DIR__, "..","..", "data", "warcraft_maps")


# ## Grid graphs

#=
For the purposes of this tutorial, we consider grid graphs, as implemented in [GridGraphs.jl](https://github.com/gdalle/GridGraphs.jl).
In such graphs, each vertex corresponds to a couple of coordinates ``(i, j)``, where ``1 \leq i \leq h`` and ``1 \leq j \leq w``.
=#

h, w = 12, 12
g = GridGraph(exp.(rand(h, w)));

#=
For convenience, `GridGraphs.jl` also provides custom functions to compute shortest paths efficiently.
Let us see what those paths look like.
=#

p = path_to_matrix(g, grid_dijkstra(g, 1, nv(g)));
spy(p)

# ## Learning options 

#=
We first need to define a few learning options: the perturbation size `ϵ`, the number of noise sample 
per dataset point `M`, the number of training epochs `nb_epochs`, the number of dataset samples `nb_samples`,
the batch size `batch_size`, and the starting learning rate `lr_start`.
=#

options = (ϵ=0.1, M=3, nb_epochs=50, nb_samples=50, batch_size = 7, lr_start = 0.001)

# ## Dataset and model

#=
As announced, we do not know the cost of each vertex, only the image of the terrains.
Let us load the dataset and keep 80% to train and 20% to test.
=#

dataset = create_dataset(decompressed_path, options.nb_samples)
train_dataset, test_dataset = train_test_split(dataset, 0.8)

#=
We can have a glimpse at a dataset point as follows:
=#
x, y_true, kwargs = test_dataset[6]
im = convert_image_for_plot(x[:,:,:,1])
plot_image_and_path(im, y_true)

#=
We can now build our embedding, a truncated Resnet18.
=#

model = create_warcraft_embedding()

# ## Train model

#= 
We now have everything to train our model.
=#

Losses, Cost_ratios = train_with_perturbed_FYL!(;
    model=model,
    train_dataset=Flux.DataLoader(train_dataset; batchsize=options.batch_size),
    test_dataset = Flux.DataLoader(test_dataset; batchsize=length(test_dataset)),
    options=options,
)

# ## Results

#=
We are interested both in the Fenchel-Young and cost ratio between true and computed shortest path.
=#
Gaps = Cost_ratios .- 1
plot_loss_and_cost_ratio(Losses, Gaps, options)

#=
To assess performance, we can compare the true and leanrned paths.
=#

θ_test =  model(x)
shortest_path = UInt8.(linear_maximizer(θ_test))

plot_image_label_path(im, shortest_path, y_true)

#=
As well as the learned weights with their true (hidden) values.
=#
plot_terrain_weights(kwargs.wg.weights, -θ_test)