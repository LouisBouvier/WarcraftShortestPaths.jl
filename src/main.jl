using WarcraftShortestPaths
using Flux

compressed_path = joinpath(@__DIR__, "..", "data", "warcraft_maps.tar.gz")
decompressed_path = joinpath(@__DIR__, "..", "data", "warcraft_maps")

options = (ϵ=1, M=10, nb_epochs=100, nb_samples=100, batch_size = 80, lr_start = 0.001)
# Construct embedding
model = create_warcraft_embedding()

# Import dataset
dataset = create_dataset(decompressed_path, options.nb_samples)
train_dataset, test_dataset = train_test_split(dataset, 0.8)

# Imitation learning
# Losses, Cost_ratios = train_with_perturbed_FYL!(;
#     model=model,
#     train_dataset=Flux.DataLoader(train_dataset; batchsize=options.batch_size),
#     test_dataset = Flux.DataLoader(test_dataset; batchsize=length(test_dataset)),
#     options=options,
# )
# # Learning by experience
Losses, Cost_ratios = train_with_perturbed_cost!(;
    model=model,
    train_dataset=Flux.DataLoader(train_dataset; batchsize=options.batch_size),
    test_dataset = Flux.DataLoader(test_dataset; batchsize=length(test_dataset)),
    options=options,
)

# Plot loss
Gaps = Cost_ratios .- 1
plot_loss_and_cost_ratio(Losses, Gaps, options)

# Eval effect
x, y_true, kwargs = test_dataset[1]
θ_test =  model(x)
shortest_path = UInt8.(linear_maximizer(θ_test))

# Display map, shortest path computed and shortest path labelled
im = convert_image_for_plot(x[:,:,:,1])
plot_image_label_path(im, shortest_path, y_true)
# plot_image_and_path(im, y_true)

# # Display labelled and computed weights
plot_terrain_weights(kwargs.wg.weights, -θ_test)