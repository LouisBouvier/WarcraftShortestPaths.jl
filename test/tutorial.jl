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
using Random
using InferOpt
using Flux 
using Graphs
using GridGraphs
using Statistics
using LinearAlgebra
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
We first need to define a few functions and options: the number of dataset samples `nb_samples`,
and the proportion of train samples `train_prop`.
=#

options = (nb_samples=100, train_prop = 0.8)

## Main functions

cost(y, θ) = dot(y, θ)
error_function(ŷ, y) = half_square_norm(ŷ - y)

function true_maximizer(θ::AbstractMatrix{R}; kwargs...) where {R<:Real}
    g = GridGraph(-θ)
    path = grid_dijkstra(g, 1, nv(g))
    y = path_to_matrix(g, path)
    return y
end

# ## Dataset and model

#=
As announced, we do not know the cost of each vertex, only the image of the terrains.
Let us load the dataset and keep 80% to train and 20% to test.
=#

data_train, data_test = generate_dataset(decompressed_path, options.nb_samples, options.train_prop)

#=
We can have a glimpse at a dataset image as follows:
=#
(X_test, Θ_test, Y_test) = data_test
x_test, θ_test_true, y_test = X_test[5], Θ_test[5], Y_test[5]
plot_map(dropdims(x_test; dims=4))

#=
The corresponding shortest path 
=#

plot_path(y_test)


#=
We can now show our embedding, a truncated Resnet18.
=#

create_warcraft_embedding()

# ## Train model

# We define a pipeline in a learning by experience setting, using a `FenchelYoungLoss`.
# You can find other pipelines in the `main.jl` file.

pipeline = (
    encoder=create_warcraft_embedding(),
    maximizer=identity,
    loss=FenchelYoungLoss(PerturbedMultiplicative(true_maximizer; ε=1., nb_samples=5)),
)


# We train over this pipeline.

(; encoder, maximizer, loss) = pipeline
pipeline_loss_imitation_y(x, θ, y) = loss(maximizer(encoder(x)), y)
apply_learning_pipeline!(
    pipeline,
    pipeline_loss_imitation_y;
    true_maximizer=true_maximizer,
    data_train=data_train,
    data_test=data_test,
    error_function=error_function,
    cost=cost,
    epochs=50,
    verbose=true,
    setting_name="paths - imitation_y",
)
# ## Results

#=
We can assess results on a test set sample.
=#

θ_test_pred =  encoder(x_test)
y_test_pred = UInt8.(true_maximizer(θ_test_pred))
plot_map(dropdims(x_test; dims=4))

#=
The true cell costs:
=#
plot_weights(θ_test_true)

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