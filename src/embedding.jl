"""
    average_tensor(x)

Average the tensor `x` along its third axis.
"""
function average_tensor(x)
    return sum(x, dims = [3])/size(x)[3]
end

"""
    permute_tensor(x)

Permute the two first axes of tensor `x`.
"""
function permute_tensor(x)
    return permutedims(x, (2,1,3,4))
end

"""
    neg_exponential_tensor(x)

Compute minus exponential element-wise on tensor `x`.
"""
function neg_exponential_tensor(x)
    return -exp.(x)
end

"""
    squeeze_last_dims(x)

Squeeze two last dimensions on tensor `x`.
"""
function squeeze_last_dims(x)
    return reshape(x, size(x)[1], size(x)[2])
end

"""
    create_warcraft_embedding()

Create and return a `Flux.Chain` embedding for the Warcraft terrains, inspired by [differentiation of blackbox combinatorial solvers](https://github.com/martius-lab/blackbox-differentiation-combinatorial-solvers/blob/master/models.py).

The embedding is made as follows:
    1) The first 5 layers of ResNet18 (convolution, batch normalization, relu, maxpooling and first resnet block).
    2) An adaptive maxpooling layer to get a (12x12x64) tensor per input image.
    3) An average over the third axis (of size 64) to get a (12x12x1) tensor per input image.
    4) The element-wize [`neg_exponential_tensor`](@ref) function to get cell weights of proper sign to apply shortest path algorithms.
    4) A squeeze function to forget the two last dimensions. 
"""
function create_warcraft_embedding()
    resnet18 = ResNet18(pretrain = false, nclasses = 1)
    model_embedding = Chain(resnet18.layers[1][1:4], 
                            AdaptiveMaxPool((12,12)), 
                            average_tensor, 
                            # permute_tensor,
                            # Flux.flatten,
                            neg_exponential_tensor, 
                            squeeze_last_dims,
                            # vec,
    )
    return model_embedding
end