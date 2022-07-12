using WarcraftShortestPaths
using InferOpt
using Flux
using Random

Random.seed!(63);
decompressed_path = joinpath(@__DIR__, "..", "data", "warcraft_maps")
options = (ϵ=1.2, M=3, nb_epochs=5, nb_samples=5, batch_size = 1, lr_start = 0.001)

## Import dataset
dataset = create_dataset(decompressed_path, options.nb_samples)
train_dataset, test_dataset = train_test_split(dataset, 0.8)

## Create learning pipeline and flux loss

# Here comes the specific InferOpt setting

pipeline =  (encoder=create_warcraft_embedding(), maximizer=identity, loss=SPOPlusLoss(true_maximizer))

# Define flux loss

(; encoder, maximizer, loss) = pipeline
# flux_loss_point(x, y, kwargs) = loss(maximizer(encoder(x)); c_true = kwargs.wg.weights)
# flux_loss_point(x, y, kwargs) = loss(maximizer(encoder(x)), y)
flux_loss_point(x, y, kwargs) = loss(maximizer(encoder(x)), -kwargs.wg.weights)
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

## Plots and save data 
# pipeline_name = "FY_Regularized/"
# options_name = "epsilon_$(options.ϵ)_M_$(options.M)_nb_epochs_$(options.nb_epochs)_nb_samples_$(options.nb_samples)_batch_size_$(options.batch_size)_lr_start_$(options.lr_start)/"
# path_to_save = "../results_article/"*pipeline_name*options_name
# mkpath(path_to_save)
# save_metrics(path = path_to_save, losses = Losses, gaps = Gaps)

# ## Plot loss and gap
# plot_loss_and_gap(Losses, Gaps, options; filepath = path_to_save*"loss_gap.pdf")

# # Eval effect
# x_test, y_test, kwargs = test_dataset[12]
# θ_test_pred =  encoder(x_test)
# y_test_pred = UInt8.(true_maximizer(θ_test_pred))

# # Display map, shortest path computed and shortest path labelled
# plot_map(dropdims(x_test; dims=4), filepath=path_to_save*"map.pdf")
# plot_weights(kwargs.wg.weights, filepath=path_to_save*"true_cost.pdf")
# plot_weights(-θ_test_pred, filepath=path_to_save*"pred_cost.pdf")
# plot_path(y_test, filepath=path_to_save*"true_path.pdf")
# plot_path(y_test_pred, filepath=path_to_save*"pred_path.pdf")
