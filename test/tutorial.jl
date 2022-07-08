# # Tutorial

# ## Context

#=
In this tutorial page we illustrate one of the possible learning pipelines for the Warcraft shortest paths problem. 
We have a sub-dataset of Warcraft terrain images and corresponding black box cost functions. 
We want to learn the cost of the cells, using a neural network embedding.
More precisely, each point in our dataset consists in:
- an image of terrain ``I``.
- a cost function ``c`` to evaluate any given path.
We don't know the true costs that were used to compute the shortest path, but we can exploit the images to approximate these costs.
The question is: how should we combine these features?
We use `InferOpt` to learn the appropriate costs, so that we may propose relevant paths in the future.
=#

using WarcraftShortestPaths
using Graphs
using GridGraphs
using Flux
using InferOpt
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

options = (ϵ=0.8, M=10, nb_epochs=50, nb_samples=100, batch_size = 20, lr_start = 0.001)

# ## Dataset and model

#=
As announced, we do not know the cost of each vertex, only the image of the terrains.
Let us load the dataset and keep 80% to train and 20% to test.
=#

dataset = create_dataset(decompressed_path, options.nb_samples)
train_dataset, test_dataset = train_test_split(dataset, 0.8)

#=
We can have a glimpse at a dataset image as follows:
=#
x_test, y_test, kwargs_test = test_dataset[12]
plot_map(dropdims(x_test; dims=4))

#=
The corresponding shortest path: 
=#

plot_path(y_test)

#=
Our embedding is a truncated Resnet18.
=#

create_warcraft_embedding()

# ## Create learning pipeline and flux loss

# Here comes the specific InferOpt setting. We learn by experience here, only based on the images and on a black box cost function.
pipeline = (
    encoder=create_warcraft_embedding(),
    maximizer=identity,
    loss=ProbabilisticComposition(
        PerturbedMultiplicative(true_maximizer; ε=1.0, nb_samples=10), cost
    )
)

# Define flux loss
(; encoder, maximizer, loss) = pipeline
flux_loss_point(x, y, kwargs) = loss(maximizer(encoder(x)); c_true = kwargs.wg.weights)
flux_loss_batch(batch) = sum(flux_loss_point(item[1], item[2], item[3]) for item in batch)

# We now have everything to train our model.
Losses, Cost_ratios = train_function!(;
    encoder=encoder,
    flux_loss = flux_loss_batch,
    train_dataset=Flux.DataLoader(train_dataset; batchsize=options.batch_size),
    test_dataset = Flux.DataLoader(test_dataset; batchsize=length(test_dataset)),
    options=options,
)

# ## Results

#=
We are interested both in the loss and cost ratio between true and computed shortest path.
=#
Gaps = Cost_ratios .- 1
plot_loss_and_gap(Losses, Gaps, options)

#=
To assess performance, we can compare the true and predicted paths.
=#

θ_test_pred =  encoder(x_test)
y_test_pred = UInt8.(true_maximizer(θ_test_pred))
plot_map(dropdims(x_test; dims=4))

#=
The true cell costs:
=#
plot_weights(kwargs_test.wg.weights)

#=
The predicted cell costs:
=#
plot_weights(-θ_test_pred)

#=
The true shortest path:
=#
plot_path(y_test)

#=
The predicted shortest path:
=#
plot_path(y_test_pred)