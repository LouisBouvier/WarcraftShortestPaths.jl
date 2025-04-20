using InferOpt
using Flux
using Flux.Optimise
using Random
using LinearAlgebra
using WarcraftShortestPaths

## SL function
function SL_training(model, data, val_data; nb_epochs=10, iterations=1, batch_size = 80, lr_start = 0.001)
    loss = FenchelYoungLoss(PerturbedMultiplicative(true_maximizer; ε=0.05, nb_samples=20))
    opt = ADAM(lr_start)
    data_train = Flux.DataLoader(data; batchsize=batch_size)

    train_costs = Float64[]
    val_costs = Float64[]
    losses = Float64[]
    params = Flux.params(model)
    best_model = deepcopy(model)
    best_episode = 0

    for epoch in 1:nb_epochs
        push!(train_costs, mean([cost(true_maximizer(model(b[1])); c_true=b[3].wg.weights) for b in data]),)
        push!(val_costs, mean([cost(true_maximizer(model(b[1])); c_true=b[3].wg.weights) for b in val_data]),)
        @info epoch, "train:", train_costs[end], "val:", val_costs[end]
        if reward_comparison(train_costs, val_costs)
            best_model = deepcopy(model)
            best_episode = epoch
        end
        # push!(costs, sum([cost(true_maximizer(model(data[j][1])); c_true=data[j][3].wg.weights) for j in 1:length(data)]) / length(data),)
        for batch in data_train
            batch_loss = 0
            gs = gradient(params) do
                batch_loss = sum([loss(model(batch[j][1]), batch[j][2]; fw_kwargs = (max_iteration=50,)) for j in 1:length(batch)])
            end
            Flux.update!(opt, params, gs)
            push!(losses, batch_loss / length(batch),)
        end
    end

    push!(train_costs, mean([cost(true_maximizer(best_model(b[1])); c_true=b[3].wg.weights) for b in data]),)
    push!(val_costs, mean([cost(true_maximizer(best_model(b[1])); c_true=b[3].wg.weights) for b in val_data]),)
    @info "final train:", train_costs[end], "final val:", val_costs[end], "best_episode:", best_episode
    return best_model, train_costs, val_costs, losses
end

## Critic function
function critic_model(x, solution, model)
    solution_expanded = kron(solution, ones(Bool, 8, 8))
    return mean(model(solution_expanded .* x))
end

function reward_comparison(train_rew, val_rew)
    means = [(train_rew[i] + val_rew[i]) / 2 for i in 1:length(train_rew)]
    last_mean = means[end]  # Mean of the last two elements
    return last_mean == minimum(means)  # Check if it's the smallest mean
end

## IL function
function IL_training(model, critic, data, val_data; nb_epochs=100, batch_size = 10, no_samples = 20, sigma_values=[0.05, 0.05], lr_values = [1e-3, 1e-3], use_critic=true, critic_steps = 0, soft=false, temp_values=[10.0, 0.1])
    loss = FenchelYoungLoss(PerturbedMultiplicative(true_maximizer; ε=0.05, nb_samples=20))
    opt_a = Optimiser(ClipValue(1e-3), Adam(lr_values[1]))
    opt_c = Optimiser(ClipValue(1e-3), Adam(lr_values[1]))
    lr_step = (lr_values[1] - lr_values[2]) / nb_epochs

    train_costs = Float64[]
    val_costs = Float64[]
    best_model = deepcopy(model)
    best_episode = 0

    prob(θ, eps) = MvNormal(θ, eps * I)
    sigma = sigma_values[1]
    sigma_step = (sigma_values[1] - sigma_values[2]) / nb_epochs
    temp = temp_values[1]
    temp_step = (temp_values[1] - temp_values[2]) / nb_epochs

    losses = Float64[]
    for epoch in 1:nb_epochs
        push!(train_costs, mean([cost(true_maximizer(model(b[1])); c_true=b[3].wg.weights) for b in data]),)
        push!(val_costs, mean([cost(true_maximizer(model(b[1])); c_true=b[3].wg.weights) for b in val_data]),)
        @info epoch, "sigma:", sigma, "lr:", opt_a.os[2].eta, "temp:", temp, "train:", train_costs[end], "val:", val_costs[end]
        if reward_comparison(train_costs, val_costs)
            best_model = deepcopy(model)
            best_episode = epoch
        end

        batches = Flux.DataLoader(data; batchsize=batch_size, shuffle=true)

        for batch in batches
            if epoch >= critic_steps
                best_solutions = []
                for b in batch
                    θ = model(b[1])
                    η = -abs.(reshape(rand(prob(reshape(θ, length(θ)), sigma), no_samples), (size(θ)..., no_samples)))
                    solutions = []
                    values = []
                    push!(solutions, true_maximizer(θ),)
                    use_critic ? push!(values, critic_model(b[1], solutions[end], critic),) : push!(values, cost(solutions[end]; c_true=b[3].wg.weights),)
                    for i in 1:no_samples
                        solution = true_maximizer(η[:, :, i])
                        push!(solutions, solution,)
                        use_critic ? push!(values, critic_model(b[1], solution, critic),) : push!(values, cost(solution; c_true=b[3].wg.weights),)
                    end
                    if soft
                        values = values ./ (-temp)
                        lse = logsumexp(values)
                        probs = exp.(values .- lse)
                        best_action = sum(probs .* solutions)
                        # any(isnan.(best_action)) ? best_action = solutions[argmax(values)] : nothing
                    else
                        best_action = solutions[argmin(values)]
                    end
                    push!(best_solutions, best_action,)
                end

                actor_loss = 0.0
                grads = gradient(Flux.params(model)) do
                    actor_loss = sum([loss(model(batch[j][1]), best_solutions[j]; fw_kwargs = (max_iteration=50,)) for j in 1:batch_size])
                end
                Flux.update!(opt_a, Flux.params(model), grads)
            end

            if use_critic
                θ = [model(batch[j][1]) for j in 1:batch_size]
                η = [-abs.(reshape(rand(prob(reshape(θ[j], length(θ[j])), sigma)), size(θ[j]))) for j in 1:batch_size]
                solutions = [true_maximizer(η[j]) for j in 1:batch_size]
                targets = [cost(solutions[j]; c_true=batch[j][3].wg.weights) for j in 1:batch_size]
                critic_inputs = [kron(solutions[j], ones(Bool, 8, 8)) .* batch[j][1] for j in 1:batch_size]
                critic_loss = 0.0
                grads = gradient(Flux.params(critic)) do
                    critic_values = [mean(critic(critic_inputs[j])) for j in 1:batch_size]
                    critic_loss = Flux.mse(critic_values, targets)
                end
                Flux.update!(opt_c, Flux.params(critic), grads)
                push!(losses, critic_loss,)
            end
        end
        sigma = max(sigma - sigma_step, sigma_values[2])
        lr = opt_a.os[2].eta
        opt_a.os[2].eta = max(lr - lr_step, lr_values[2])
        opt_c.os[2].eta = max(lr - lr_step, lr_values[2])
        temp = max(temp - temp_step, temp_values[2])
    end

    push!(train_costs, mean([cost(true_maximizer(best_model(b[1])); c_true=b[3].wg.weights) for b in data]),)
    push!(val_costs, mean([cost(true_maximizer(best_model(b[1])); c_true=b[3].wg.weights) for b in val_data]),)
    @info "final train:", train_costs[end], "final val:", val_costs[end], "best_episode:", best_episode
    return best_model, train_costs, val_costs, losses
end

function PPO_training(model, critic, data, val_data; nb_epochs=100, batch_size = 4, clip = 0.2, sigma_values=[0.05, 0.05], lr_values = [1e-3, 1e-3], use_critic=true, critic_steps = 0)
    opt_a = Optimiser(ClipValue(1e-3), Adam(lr_values[1]))
    opt_c = Optimiser(ClipValue(1e-3), Adam(lr_values[1]))
    lr_step = (lr_values[1] - lr_values[2]) / nb_epochs

    train_costs = Float64[]
    val_costs = Float64[]
    best_model = deepcopy(model)
    best_episode = 0

    prob(θ, eps) = MvNormal(θ, eps * I)
    sigma = sigma_values[1]
    sigma_step = (sigma_values[1] - sigma_values[2]) / nb_epochs
    sigma_avg = ((sigma_values[1] + sigma_values[2]) / 2) * 1

    losses = Float64[]
    for epoch in 1:nb_epochs
        push!(train_costs, mean([cost(true_maximizer(model(b[1])); c_true=b[3].wg.weights) for b in data]),)
        push!(val_costs, mean([cost(true_maximizer(model(b[1])); c_true=b[3].wg.weights) for b in val_data]),)
        @info epoch, "sigma:", sigma, "lr", opt_a.os[2].eta, "train:", train_costs[end], "val:", val_costs[end]
        if reward_comparison(train_costs, val_costs)
            best_model = deepcopy(model)
            best_episode = epoch
        end

        batches = Flux.DataLoader(data; batchsize=batch_size, shuffle=true)

        for batch in batches
            if epoch >= critic_steps
                thetas = []
                etas = []
                advantages = []
                for b in batch
                    push!(thetas, model(b[1]),)
                    push!(etas, -abs.(reshape(rand(prob(reshape(thetas[end], length(thetas[end])), sigma)), size(thetas[end]))),)
                    use_critic ? push!(advantages, critic_model(b[1], true_maximizer(thetas[end]), critic) - critic_model(b[1], true_maximizer(etas[end]), critic)) : push!(advantages, cost(true_maximizer(thetas[end]); c_true=b[3].wg.weights) - cost(true_maximizer(etas[end]); c_true=b[3].wg.weights))
                end
            
                actor_loss = 0.0
                grads = gradient(Flux.params(model)) do
                    old_probs = [pdf(prob(reshape(thetas[b], length(thetas[b])), sigma_avg), reshape(etas[b], length(etas[b]))) for b in 1:batch_size]
                    new_probs = [pdf(prob(reshape(model(batch[b][1]), length(thetas[b])), sigma_avg), reshape(etas[b], length(etas[b]))) for b in 1:batch_size]
                    ratio_unclipped = [new_probs[b] / old_probs[b] for b in 1:batch_size]
                    ratio_clipped = clamp.(ratio_unclipped, 1-clip, 1+clip)
                    actor_loss = -mean(min.(ratio_unclipped .* advantages, ratio_clipped .* advantages)) # return?
                end
                Flux.update!(opt_a, Flux.params(model), grads)
                push!(losses, actor_loss,)
            end

            if use_critic
                θ = [model(batch[j][1]) for j in 1:batch_size]
                η = [-abs.(reshape(rand(prob(reshape(θ[j], length(θ[j])), sigma)), size(θ[j]))) for j in 1:batch_size]
                solutions = [true_maximizer(η[j]) for j in 1:batch_size]
                targets = [cost(solutions[j]; c_true=batch[j][3].wg.weights) for j in 1:batch_size]
                critic_inputs = [kron(solutions[j], ones(Bool, 8, 8)) .* batch[j][1] for j in 1:batch_size]
                critic_loss = 0.0
                grads = gradient(Flux.params(critic)) do
                    critic_values = [mean(critic(critic_inputs[j])) for j in 1:batch_size]
                    critic_loss = Flux.mse(critic_values, targets)
                end
                Flux.update!(opt_c, Flux.params(critic), grads)
            end
        end
        sigma = max(sigma - sigma_step, sigma_values[2])
        lr = opt_a.os[2].eta
        opt_a.os[2].eta = max(lr - lr_step, lr_values[2])
        opt_c.os[2].eta = max(lr - lr_step, lr_values[2])
    end

    push!(train_costs, mean([cost(true_maximizer(best_model(b[1])); c_true=b[3].wg.weights) for b in data]),)
    push!(val_costs, mean([cost(true_maximizer(best_model(b[1])); c_true=b[3].wg.weights) for b in val_data]),)
    @info "final train:", train_costs[end], "final val:", val_costs[end], "best_episode:", best_episode
    return best_model, train_costs, val_costs, losses
end