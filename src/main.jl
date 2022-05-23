using WarcraftShortestPath
using Flux

compressed_path = joinpath(@__DIR__, "..", "data", "warcraft_maps.tar.gz")
decompressed_path = joinpath(@__DIR__, "..", "data", "warcraft_maps")

options = (ϵ=2.0, M=5, nb_epochs=50, nb_samples=50, batch_size = 7, lr_start = 0.001)
# Construct embedding
model = create_warcraft_embedding()

# Import dataset
dataset = create_dataset(decompressed_path, options.nb_samples)
train_dataset, test_dataset = train_test_split(dataset, 0.8)

# Imitation learning
Losses, Cost_ratios = train_with_perturbed_FYL!(;
    model=model,
    train_dataset=Flux.DataLoader(train_dataset; batchsize=options.batch_size),
    test_dataset = Flux.DataLoader(test_dataset; batchsize=length(test_dataset)),
    options=options,
)
# Learning by experience
Losses, Cost_ratios = train_with_perturbed_cost!(;
    model=model,
    train_dataset=Flux.DataLoader(train_dataset; batchsize=options.batch_size),
    test_dataset = Flux.DataLoader(test_dataset; batchsize=length(test_dataset)),
    options=options,
)

# # Plot loss
plot_loss_and_cost_ratio(Losses, Cost_ratios, options)
# plot_loss(Losses, options)

# # # # Eval effect
# x, y_true, kwargs = test_dataset[6]
# θ_test =  model(x)
# shortest_path = warcraft_shortest_path(θ_test; kwargs...)
# # θ_true = grid_to_vector(kwargs.wg.cell_costs)
# # shortest_path = maximizer(-θ_true; kwargs...)

# sum(y_true[i]*grid_to_vector(kwargs.wg.cell_costs)[i] for i = 1:144)
# sum(shortest_path[i]*grid_to_vector(kwargs.wg.cell_costs)[i] for i = 1:144)

# grid_true = vector_to_grid(y_true)
# grid_computed = vector_to_grid(shortest_path)
# im = convert_image_for_plot(x[:,:,:,1])
# plot_image_label_path(im, grid_computed, grid_true)


# plot_terrain_weights(kwargs.wg.cell_costs, -vector_to_grid(θ_test))