"""
    convert_image_for_plot(image::Array{Float32,3})::Array{RGB{N0f8},2}

Convert `image` to the proper data format to enable plots in Julia.
"""
function convert_image_for_plot(image::Array{Float32,3})::Array{RGB{N0f8},2}
    new_img = Array{RGB{N0f8},2}(undef, size(image)[1], size(image)[2])
    for i = 1:size(image)[1]
        for j = 1:size(image)[2]
            new_img[i,j] = RGB{N0f8}(image[i,j,1], image[i,j,2], image[i,j,3])
        end
    end
    return new_img
end

"""
    plot_image_and_path(im::Array{RGB{N0f8}, 2}, zero_one_path::Matrix{UInt8})

Plot the image `im` and the path `zero_one_path` on the same Figure.
"""
function plot_image_and_path(im::Array{RGB{N0f8}, 2}, zero_one_path::Matrix{UInt8})
    p1 = plot(im, title = "Terrain map", ticks = nothing, border = nothing)
    p2 = plot(Gray.(zero_one_path), title = "Path", ticks = nothing, border = nothing)
    plot(p1, p2, layout = (1, 2))
end

"""
    plot_image_label_path(im::Array{RGB{N0f8}, 2}, zero_one_path::Matrix{UInt8}, label::Matrix{UInt8})

Plot the image `im`, the path `zero_one_path` and the labelled path `label` on the same Figure.
"""
function plot_image_label_path(im::Array{RGB{N0f8}, 2}, zero_one_path::Matrix{UInt8}, label::Matrix{UInt8})
    p1 = plot(im, title = "Terrain map", ticks = nothing, border = nothing)
    p2 = plot(Gray.(zero_one_path), title = "Path computed", ticks = nothing, border = nothing)
    p3 = plot(Gray.(label), title = "Path label", ticks = nothing, border = nothing)
    plot(p1, p2, p3, layout = (1, 3))
end

"""
    plot_loss(losses::Matrix{Float64}, options::NamedTuple)

Plot the train and test losses computed over epochs.
"""
function plot_loss(losses::Matrix{Float64}, options::NamedTuple)
    x = collect(1:options.nb_epochs)
    plot(x, losses, title = "Loss", xlabel = "epochs", ylabel = "loss", label = ["train" "test"])
end

"""
    plot_loss_and_cost_ratio(losses::Matrix{Float64}, gaps::Matrix{Float64},  options::NamedTuple)

Plot the train and test losses, as well as the train and test gaps computed over epochs.
"""
function plot_loss_and_cost_ratio(losses::Matrix{Float64}, gaps::Matrix{Float64},  options::NamedTuple)
    x = collect(1:options.nb_epochs)
    p1 = plot(x, losses, title = "Loss", xlabel = "epochs", ylabel = "loss", label = ["train" "test"])
    p2 = plot(x, gaps, title = "Gap", xlabel = "epochs", ylabel = "ratio", label = ["train" "test"])
    plot(p1, p2, layout = (1, 2))
end

"""
    plot_terrain_weights(weights_label::Matrix{Float16}, computed_labels::Matrix{Float32})

Plot both the cell costs labelled and computed on the same colormap Figure.
"""
function plot_terrain_weights(weights_label::Matrix{Float16}, computed_labels::Matrix{Float32})
    p1 = heatmap(weights_label, title = "Label weights", ticks = nothing, border = nothing, yflip = true, aspect_ratio=:equal)
    p2 = heatmap(computed_labels, title = "Computed weights", ticks = nothing, border = nothing, yflip = true, aspect_ratio=:equal)
    plot(p1, p2, layout = (1, 2))
end