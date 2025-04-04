"""
    average_tensor(x)

Average the tensor `x` along its third axis.
"""
function average_tensor(x)
    return sum(x, dims = [3])/size(x)[3]
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
    resnet18 = ResNet(18, pretrain = false, nclasses = 1)
    model_embedding = Chain(resnet18.layers[1][1:3], # originally: resnet18.layers[1][1:4]
                            AdaptiveMaxPool((12,12)), 
                            average_tensor, 
                            neg_exponential_tensor, 
                            squeeze_last_dims,
    )
    return model_embedding
end

function new_warcraft_embedding()
    return Chain(Conv((8, 8), 3 => 64, pad=3, stride=2, bias=false, tanh),
                MaxPool((3, 3), pad=1, stride=2),
                Conv((3, 3), 64 => 64, pad=1, bias=false, relu),
                Conv((3, 3), 64 => 64, pad=1, bias=false, tanh),
                Conv((3, 3), 64 => 32, pad=1, bias=false, relu),
                Conv((3, 3), 32 => 16, pad=1, bias=false, tanh),
                Conv((3, 3), 16 => 8, pad=1, bias=false, relu),
                AdaptiveMeanPool((12, 12)), average_tensor, neg_exponential_tensor, squeeze_last_dims
                )
end

function critic_warcraft_embedding()
    return Chain(Conv((8, 8), 3 => 64, pad=3, stride=2, bias=false, tanh),
                MeanPool((3, 3), pad=1, stride=2),
                Conv((3, 3), 64 => 64, pad=1, bias=false, relu),
                Conv((3, 3), 64 => 64, pad=1, bias=false, tanh),
                Conv((3, 3), 64 => 32, pad=1, bias=false, relu),
                Conv((3, 3), 32 => 16, pad=1, bias=false, tanh),
                Conv((3, 3), 16 => 8, pad=1, bias=false, relu),
                AdaptiveMeanPool((12, 12)), average_tensor, neg_exponential_tensor, squeeze_last_dims,
                vec, Dense(144, 40, tanh), Dense(40, 10, relu), Dense(10, 1)
    )
end