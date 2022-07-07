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


function plot_map(map_matrix::Array{<:Real,3}; filepath=nothing)
    img = convert_image_for_plot(map_matrix)
    pl = Plots.plot(
        img;
        aspect_ratio=:equal,
        framestyle=:none,
        size=(500, 500)
    )
    isnothing(filepath) || Plots.savefig(pl, filepath)
    return pl
end

function plot_weights(weights::Matrix{<:Real}; filepath=nothing)
    pl = Plots.heatmap(
        weights;
        yflip=true,
        aspect_ratio=:equal,
        framestyle=:none,
        padding=(0., 0.),
        size=(500, 500)
    )
    isnothing(filepath) || Plots.savefig(pl, filepath)
    return pl
end

function plot_path(path::Matrix{<:Integer}; filepath=nothing)
    pl = Plots.plot(
        Gray.(path .* 0.7);
        aspect_ratio=:equal,
        framestyle=:none,
        size=(500, 500)
    )
    isnothing(filepath) || Plots.savefig(pl, filepath)
    return pl
end