using InferOpt
using Flux
using Random
using WarcraftShortestPaths

Random.seed!(63);
decompressed_path = joinpath(@__DIR__, "..", "data", "warcraft_maps")
options = (ϵ=0.05, M=20, nb_epochs=100, dataset_size=100, batch_size = 80, lr_start = 0.001)

## Import dataset
dataset = create_dataset(decompressed_path, options.dataset_size)
train_dataset, test_dataset = train_test_split(dataset, 0.8)

## Create learning pipeline and flux loss
# Here comes the specific InferOpt setting

pipeline = (
    encoder=create_warcraft_embedding(),
    maximizer=identity,
    loss=FenchelYoungLoss(PerturbedMultiplicative(true_maximizer; ε=options.ϵ, nb_samples=options.M)),
)

# Define flux loss
(; encoder, maximizer, loss) = pipeline
# flux_loss_point(x, y, kwargs) = loss(maximizer(encoder(x)); c_true = kwargs.wg.weights, fw_kwargs = (max_iteration=100,))
flux_loss_point(x, y, kwargs) = loss(maximizer(encoder(x)), y; fw_kwargs = (max_iteration=50,))
# flux_loss_point(x, y, kwargs) = loss(maximizer(encoder(x)), -kwargs.wg.weights)
flux_loss_batch(batch) = sum(flux_loss_point(item[1], item[2], item[3]) for item in batch)

## Training function
Losses, Cost_ratios = train_function!(;
    encoder=encoder,
    flux_loss = flux_loss_batch,
    train_dataset=Flux.DataLoader(train_dataset; batchsize=options.batch_size),
    test_dataset = Flux.DataLoader(test_dataset; batchsize=length(test_dataset)),
    options=options,
)
Gaps = Cost_ratios .- 1

## Plot loss and gap
display(plot_loss_and_gap(Losses, Gaps, options))

# Eval effect
x_test, y_test, kwargs = test_dataset[12]
θ_test_pred =  encoder(x_test)
y_test_pred = UInt8.(true_maximizer(θ_test_pred))

# Display map, shortest path computed and shortest path labelled
display(plot_map(dropdims(x_test; dims=4)))
display(plot_weights(kwargs.wg.weights))
display(plot_weights(-θ_test_pred))
display(plot_path(y_test))
display(plot_path(y_test_pred))
