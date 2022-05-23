"""
    shortest_path_cost_ratio(model, x, y, kwargs)

Compute the ratio between the cost of the solution given by the `model` cell costs and the cost of the true solution.

We evaluate both the shortest path with respect to the weights given by `model(x)` and the labelled shortest path `y`
using the true cell costs stored in `kwargs.wg.cell_costs`. 
This ratio is by definition greater than one. The closer it is to one, the better is the solution given by the current 
weights of `model`. We thus track this metric during training.
"""
function shortest_path_cost_ratio(model, x, y, kwargs)
    true_cell_costs = grid_to_vector(kwargs.wg.cell_costs)
    θ_computed = model(x)
    shortest_path_computed = warcraft_shortest_path(θ_computed, wg = kwargs.wg)
    return sum(true_cell_costs[i]*shortest_path_computed[i] for i = 1:144)/sum(y[i]*true_cell_costs[i] for i=1:144)
end

"""
    shortest_path_cost_ratio(model, batch)

Compute the average cost ratio between computed and true shorest paths over `batch`. 
"""
function shortest_path_cost_ratio(model, batch)
    return sum(shortest_path_cost_ratio(model, item[1], item[2], item[3]) for item in batch)/length(batch)
end

"""
    shortest_path_cost_ratio(;model, dataset)

Compute the average cost ratio between computed and true shorest paths over `dataset`. 
"""
function shortest_path_cost_ratio(;model, dataset)
    return sum(shortest_path_cost_ratio(model, batch) for batch in dataset)/length(dataset)
end

"""
    train_with_perturbed_FYL!(;model::Flux.Chain, train_dataset, test_dataset, options::NamedTuple)

Train `model` over `train_dataset` and test on `test_dataset` with perturbed Fenchel Young loss. 

This training involves differentiation through argmax with perturbed maximizers, using [InferOpt](https://github.com/axelparmentier/InferOpt.jl) package.
The task is to learn the best parameters for `model`, so that when solving the shortest path problem with its output cell costs, the 
given solution is close to the labelled shortest path corresponding to the input Warcraft terrain image.
Hyperparameters are passed with `options`. During training, the average train and test losses are stored, as well as the average 
cost ratio computed with [`shortest_path_cost_ratio`](@ref) both on the train and test datasets.
"""
function train_with_perturbed_FYL!(;model::Flux.Chain, train_dataset, test_dataset, options::NamedTuple)
    # Store the train loss
    losses = Matrix{Float64}(undef, options.nb_epochs, 2)
    cost_ratios = Matrix{Float64}(undef, options.nb_epochs, 2)
    # Define model and loss
    loss = FenchelYoungLoss(Perturbed(warcraft_shortest_path, options.ϵ, options.M))
    # Optimizer
    opt = ADAM(options.lr_start)
    # Pipeline
    flux_loss(x, y, kwargs) = loss(model(x), y; wg=kwargs.wg)
    flux_loss(batch) = sum(flux_loss(item[1], item[2], item[3]) for item in batch)
    # model parameters
    par = Flux.params(model)
    # Train loop
    @showprogress "Training epoch: " for epoch in 1:options.nb_epochs
        for batch in train_dataset
            batch_loss = 0
            gs = gradient(par) do
                batch_loss = flux_loss(batch)
            end
            losses[epoch, 1] += batch_loss
            Flux.update!(opt, par, gs)
        end
        losses[epoch, 1] = losses[epoch, 1]/(length(train_dataset)*options.batch_size)
        losses[epoch, 2] = sum([flux_loss(batch) for batch in test_dataset])/(length(test_dataset)*options.batch_size)
        cost_ratios[epoch, 1] = shortest_path_cost_ratio(model = model, dataset = train_dataset)
        cost_ratios[epoch, 2] = shortest_path_cost_ratio(model = model, dataset = test_dataset)
    end
    return losses, cost_ratios
end


"""
    train_with_perturbed_cost!(;model::Flux.Chain, train_dataset, test_dataset, options::NamedTuple)

Train `model` over `train_dataset` and test on `test_dataset` with perturbed cost. 

This training involves differentiation through argmax with perturbed cost, using [InferOpt](https://github.com/axelparmentier/InferOpt.jl) package.
The task is to learn the best parameters for `model`, so that when solving the shortest path problem with its output cell costs, the 
given solution has a low cost with respect to the true cell weights labelled.
Hyperparameters are passed with `options`. During training, the average train and test losses are stored, as well as the average 
cost ratio computed with [`shortest_path_cost_ratio`](@ref) both on the train and test datasets.
"""
function train_with_perturbed_cost!(;model::Flux.Chain, train_dataset, test_dataset, options::NamedTuple)
    # Store the train loss
    losses = Matrix{Float64}(undef, options.nb_epochs, 2)
    cost_ratios = Matrix{Float64}(undef, options.nb_epochs, 2)
    # Define regularized pred
    regpred = Perturbed(warcraft_shortest_path; ε=options.ϵ, M=options.M)
    # Define cost 
    cost(x, kwargs) = dot(regpred(model(x); wg=kwargs.wg), -Flux.flatten(permutedims(kwargs.wg.cell_costs, (2,1))))
    cost(batch) = sum(cost(item[1], item[3]) for item in batch)
    # Optimizer
    opt = ADAM(options.lr_start)
    # model parameters
    par = Flux.params(model)
    # Train loop
    @showprogress "Training epoch: " for epoch in 1:options.nb_epochs
        for batch in train_dataset
            batch_loss = 0
            gs = gradient(par) do
                batch_loss = cost(batch)
            end
            losses[epoch, 1] += batch_loss
            Flux.update!(opt, par, gs)
        end
        losses[epoch, 1] = losses[epoch, 1]/(length(train_dataset)*options.batch_size)
        losses[epoch, 2] = sum([cost(batch) for batch in test_dataset])/(length(test_dataset)*options.batch_size)
        cost_ratios[epoch, 1] = shortest_path_cost_ratio(model = model, dataset = train_dataset)
        cost_ratios[epoch, 2] = shortest_path_cost_ratio(model = model, dataset = test_dataset)
    end
    return losses, cost_ratios
end