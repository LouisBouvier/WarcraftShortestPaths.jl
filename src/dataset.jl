"""
    decompress_dataset(compressed_path::String, decompressed_path::String)

Decompress the dataset located at `compressed_path` and save it at `decompressed_path`.
"""
function decompress_dataset(compressed_path::String, decompressed_path::String)
    GZip.open(compressed_path, "r") do tarball
        Tar.extract(tarball, decompressed_path)
    end
end

"""
    read_dataset(decompressed_path::String, dtype::String="train")

Read the dataset of type `dtype` at the `decompressed_path` location.

The dataset is made of images of Warcraft terrains, cell cost labels and shortest path labels.
They are returned separately, with proper axis permutation and image scaling to be consistent with 
`Flux` embeddings.
"""
function read_dataset(decompressed_path::String, dtype::String="train")
    # Open files
    data_dir = joinpath(decompressed_path, "warcraft_shortest_path_oneskin", "12x12")
    data_suffix = "maps"
    terrain_images = npzread(joinpath(data_dir, dtype * "_" * data_suffix * ".npy"))
    terrain_weights = npzread(joinpath(data_dir, dtype * "_vertex_weights.npy"))
    terrain_labels = npzread(joinpath(data_dir, dtype * "_shortest_paths.npy"))
    # Reshape for Flux
    terrain_images = permutedims(terrain_images, (2, 3, 4, 1))
    terrain_labels = permutedims(terrain_labels, (2, 3, 1))
    terrain_weights = permutedims(terrain_weights, (2, 3, 1))
    # Normalize images
    terrain_images = Array{Float32}(terrain_images ./ 255)
    println("Train images shape: ", size(terrain_images))
    println("Train labels shape: ", size(terrain_labels))
    println("Weights shape:", size(terrain_weights))
    return terrain_images, terrain_labels, terrain_weights
end

"""
    create_dataset(decompressed_path::String, nb_samples::Int=10000)

Create the dataset corresponding to the data located at `decompressed_path`, possibly sub-sampling `nb_samples` points.

The dataset is made of images of Warcraft terrains, cell cost labels and shortest path labels.
It is a `Vector` of tuples, each `Tuple` being a dataset point.
"""
function create_dataset(decompressed_path::String, nb_samples::Int=10000)
    terrain_images, terrain_labels, terrain_weights = read_dataset(
        decompressed_path, "train"
    )
    X = [
        reshape(terrain_images[:, :, :, i], (size(terrain_images[:, :, :, i])..., 1)) for
        i in 1:nb_samples
    ]
    Y = [terrain_labels[:, :, i] for i in 1:nb_samples]
    WG = [(wg=GridGraph(terrain_weights[:, :, i]),) for i in 1:nb_samples]
    return collect(zip(X, Y, WG))
end

"""
    train_test_split(X::AbstractVector, train_percentage::Real=0.5)

Split a dataset contained in `X` into train and test datasets.

The proportion of the initial dataset kept in the train set is `train_percentage`.
"""
function train_test_split(X::AbstractVector, train_percentage::Real=0.5)
    N = length(X)
    N_train = floor(Int, N * train_percentage)
    N_test = N - N_train
    train_ind, test_ind = 1:N_train, (N_train + 1):(N_train + N_test)
    X_train, X_test = X[train_ind], X[test_ind]
    return X_train, X_test
end