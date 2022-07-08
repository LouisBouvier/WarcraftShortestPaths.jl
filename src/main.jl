using WarcraftShortestPaths
using InferOpt
using Flux

compressed_path = joinpath(@__DIR__, "..", "data", "warcraft_maps.tar.gz")
decompressed_path = joinpath(@__DIR__, "..", "data", "warcraft_maps")

options = (ϵ=0.8, M=10, nb_epochs=50, nb_samples=100, batch_size = 20, lr_start = 0.001)

## Import dataset
dataset = create_dataset(decompressed_path, options.nb_samples)
train_dataset, test_dataset = train_test_split(dataset, 0.8)

## Create learning pipeline and flux loss

# Here comes the specific InferOpt setting

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

## Training function
Losses, Cost_ratios = train_function!(;
    encoder=encoder,
    flux_loss = flux_loss_batch,
    train_dataset=Flux.DataLoader(train_dataset; batchsize=options.batch_size),
    test_dataset = Flux.DataLoader(test_dataset; batchsize=length(test_dataset)),
    options=options,
)

## Plot loss and gap
Gaps = Cost_ratios .- 1
plot_loss_and_gap(Losses, Gaps, options)

# # Eval effect
# x_test, y_test, kwargs = test_dataset[12]
# θ_test_pred =  encoder(x_test)
# y_test_pred = UInt8.(true_maximizer(θ_test_pred))

# # Display map, shortest path computed and shortest path labelled
# plot_map(dropdims(x_test; dims=4), filepath="map.pdf")
# plot_weights(kwargs.wg.weights, filepath="true_cost.pdf")
# plot_weights(-θ_test_pred, filepath="pred_cost.pdf")
# plot_path(y_test, filepath="true_path.pdf")
# plot_path(y_test_pred, filepath="pred_path.pdf")