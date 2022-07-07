using WarcraftShortestPaths
using Random
using InferOpt
using Flux 
using Graphs
using GridGraphs
using Statistics
using LinearAlgebra


Random.seed!(63)

decompressed_path = joinpath(@__DIR__, "..", "data", "warcraft_maps")
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

## Pipelines

pipelines_imitation_θ = [
    # SPO+
    (encoder=create_warcraft_embedding(), maximizer=identity, loss=SPOPlusLoss(true_maximizer)),
]

pipelines_imitation_y = [
    # Interpolation  # TODO: make it work
    # (
    #     encoder=encoder_factory(),
    #     maximizer=Interpolation(true_maximizer; λ=5.0),
    #     loss=Flux.Losses.mse,
    # ),
    # Perturbed + FYL
    (
        encoder=create_warcraft_embedding(),
        maximizer=identity,
        loss=FenchelYoungLoss(PerturbedMultiplicative(true_maximizer; ε=0.7, nb_samples=7)),
    ),
    # Perturbed + other loss
    (
        encoder=create_warcraft_embedding(),
        maximizer=PerturbedMultiplicative(true_maximizer; ε=1.0, nb_samples=10),
        loss=Flux.Losses.mse,
    ),
    # Generic regularized + FYL
    (
        encoder=create_warcraft_embedding(),
        maximizer=identity,
        loss=FenchelYoungLoss(
            RegularizedGeneric(true_maximizer, half_square_norm, identity)
        ),
    ),
    # Generic regularized + other loss
    (
        encoder=create_warcraft_embedding(),
        maximizer=RegularizedGeneric(true_maximizer, half_square_norm, identity),
        loss=Flux.Losses.mse,
    ),
]

pipelines_experience = [
    (
        encoder=create_warcraft_embedding(),
        maximizer=identity,
        loss=ProbabilisticComposition(
            PerturbedMultiplicative(true_maximizer; ε=1.0, nb_samples=10), cost
        ),
    ),
    (
        encoder=create_warcraft_embedding(),
        maximizer=identity,
        loss=ProbabilisticComposition(
            RegularizedGeneric(true_maximizer, half_square_norm, identity), cost
        ),
    ),
]

## Dataset generation

data_train, data_test = generate_dataset(decompressed_path, options.nb_samples, options.train_prop)

## Test loop

# for pipeline in pipelines_imitation_θ
#     pipeline = deepcopy(pipeline)
#     (; encoder, maximizer, loss) = pipeline
#     pipeline_loss_imitation_θ(x, θ, y) = loss(maximizer(encoder(x)), θ)
#     apply_learning_pipeline!(
#         pipeline,
#         pipeline_loss_imitation_θ;
#         maximizer=maximizer,
#         data_train=data_train,
#         data_test=data_test,
#         error_function=error_function,
#         cost=cost,
#         epochs=100,
#         verbose=true,
#         setting_name="paths - imitation_θ",
#     )
# end

pipeline = deepcopy(pipelines_imitation_y[1])
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

# for pipeline in pipelines_experience
#     pipeline = deepcopy(pipeline)
#     (; encoder, maximizer, loss) = pipeline
#     pipeline_loss_experience(x, θ, y) = loss(maximizer(encoder(x)); instance=x)
#     apply_learning_pipeline!(
#         pipeline,
#         pipeline_loss_experience;
#         maximizer=maximizer,
#         data_train=data_train,
#         data_test=data_test,
#         error_function=error_function,
#         cost=cost,
#         epochs=1000,
#         verbose=true,
#         setting_name="paths - experience",
#     )
# end


## visuals 

(X_test, Θ_test, Y_test) = data_test
x_test, θ_test_true, y_test = X_test[4], Θ_test[4], Y_test[4]

θ_test_pred =  encoder(x_test)
y_test_pred = UInt8.(true_maximizer(θ_test_pred))

# Display map, shortest path computed and shortest path labelled
plot_map(dropdims(x_test; dims=4), filepath="map.pdf")
plot_weights(θ_test_true, filepath="true_cost.pdf")
plot_weights(-θ_test_pred, filepath="pred_cost.pdf")
plot_path(y_test, filepath="true_path.pdf")
plot_path(y_test_pred, filepath="pred_path.pdf")