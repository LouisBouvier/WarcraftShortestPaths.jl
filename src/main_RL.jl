using InferOpt
using Flux
using Flux.Optimise
using Random
using StatsBase
using Distributions
using LinearAlgebra
using WarcraftShortestPaths
using JLD2
using Plots

includet("RL_algorithms.jl")

## Import dataset
Random.seed!(63);
data_path = joinpath(@__DIR__, "..", "data");
options = (nb_epochs=100, batch_size = 80, lr_start = 0.001);

dataset = create_dataset(data_path, 200);
train_dataset, val_dataset, test_dataset = train_test_split(dataset, 0.6; val_percentage=0.2, use_val=true);

## Anticipative and heuristic solutions
anticipative_train = mean([cost(y; c_true=kwargs.wg.weights) for (x, y, kwargs) in train_dataset]) # 30.44

greedy_path = I(12)
greedy_train = mean([cost(greedy_path; c_true=kwargs.wg.weights) for (x, y, kwargs) in train_dataset]) # 43.1

model_random = new_warcraft_embedding();
random_train = mean([cost(true_maximizer(model_random(x)); c_true=kwargs.wg.weights) for (x, y, kwargs) in train_dataset]) # 43.1

## SL solution
Random.seed!(0);
model_SL = new_warcraft_embedding();
model_SL, train_SL, val_SL, losses_SL = SL_training(model_SL, train_dataset, val_dataset; nb_epochs=200, batch_size=60, lr_start=0.001)
sl_runtime = @timed SL_training(model_SL, train_dataset, val_dataset; nb_epochs=200, batch_size=60, lr_start=0.001)
sl_runtime.time
SL_train = mean([cost(true_maximizer(model_SL(x)); c_true=kwargs.wg.weights) for (x, y, kwargs) in train_dataset]) # 30.44
fig = plot(train_SL, label="train history"; marker=:o)
plot!(fig, val_SL, label="val history"; marker=:o)
SL_final_train_rew = [cost(true_maximizer(model_SL(x)); c_true=kwargs.wg.weights) for (x, y, kwargs) in train_dataset];
SL_final_test_rew = [cost(true_maximizer(model_SL(x)); c_true=kwargs.wg.weights) for (x, y, kwargs) in test_dataset];
jldsave("logs/wcsp_sl_best_model.jld2"; model=model_SL, train_rew=train_SL, val_rew=val_SL, train_final=SL_final_train_rew, test_final=SL_final_test_rew)

function sl_runs(seeds)
    all_results = []
    for s in seeds
        Random.seed!(s)
        model_SL = new_warcraft_embedding()
        model_SL, train_SL, val_SL, losses_SL = SL_training(model_SL, train_dataset, val_dataset; nb_epochs=200, batch_size=60, lr_start=0.001)
        final_tr = [cost(true_maximizer(model_SL(x)); c_true=kwargs.wg.weights) for (x, y, kwargs) in train_dataset]
        final_te = [cost(true_maximizer(model_SL(x)); c_true=kwargs.wg.weights) for (x, y, kwargs) in test_dataset]
        push!(all_results, (seed=s, model=model_SL, train_rew=train_SL, val_rew=val_SL, train_final=final_tr, test_final=final_te))
    end
    return all_results
end

sl_results = sl_runs([1, 2, 3, 4, 5, 6, 7, 8, 9])
jldsave("logs/wcsp_sl_random_seeds.jld2"; results=sl_results)

## IL solution
Random.seed!(0);
critic_IL = critic_warcraft_embedding();
model_IL = new_warcraft_embedding();
model_IL, train_IL, val_IL, losses_IL = IL_training(model_IL, critic_IL, train_dataset, val_dataset;
    nb_epochs = 200, batch_size = 60, no_samples = 80, sigma_values = [0.05, 0.02], lr_values = [1e-3, 5e-4], use_critic = false, critic_steps = 0, soft=true, temp_values=[1e1, 1e-1]
)
IL_train = mean([cost(true_maximizer(model_IL(x)); c_true=kwargs.wg.weights) for (x, y, kwargs) in train_dataset]) # 30.47
il_runtime = @timed IL_training(model_IL, critic_IL, train_dataset, val_dataset; nb_epochs = 200, batch_size = 60, no_samples = 80, sigma_values = [0.05, 0.02], lr_values = [1e-3, 5e-4], use_critic = false, critic_steps = 0, soft=true, temp_values=[1e1, 1e-1])
il_runtime.time
# temp: 1e-2, 1e-1, 1e0, 1e1, 1e2

function IL_test(sigma_steps, lr_steps, temp_steps, seeds; soft=true)
    final_train = []
    final_val = []
    all_rews = []
    for i in sigma_steps
        for j in lr_steps
            for k in temp_steps
                for s in seeds
                    Random.seed!(s);
                    model_IL = new_warcraft_embedding();
                    model_IL, train_IL, val_IL, losses_IL = IL_training(model_IL, critic_IL, train_dataset, val_dataset;
                        nb_epochs = 200, batch_size = 60, no_samples = 80, sigma_values = i, lr_values = j, use_critic = false, critic_steps = 0, soft=true, temp_values=k
                    )
                    final_tr = [cost(true_maximizer(model_IL(x)); c_true=kwargs.wg.weights) for (x, y, kwargs) in train_dataset]
                    final_te = [cost(true_maximizer(model_IL(x)); c_true=kwargs.wg.weights) for (x, y, kwargs) in test_dataset]
                    push!(all_rews, (sigma=i, lr=j, temp=k, seed=s, model=deepcopy(model_IL), train_rew=train_IL, val_rew=val_IL, train_final=final_tr, test_final=final_te))
                    push!(final_train, train_IL[end])
                    push!(final_val, val_IL[end])
                end
            end
        end
    end
    return final_train, final_val, all_rews
end

train_rews, val_rews, all_rews = IL_test(
    [[0.1, 0.05], [0.05, 0.02], [0.02, 0.005]],
    [[2e-3, 1e-3], [1e-3, 5e-4], [5e-4, 1e-4]],
    [[1e0, 1e-1], [1e-1, 1e-2]],
    [0]
)
train_rews, val_rews, il_results = IL_test([[0.1, 0.05]], [[0.002, 0.001]], [[0.1, 0.01]], [1, 2, 3, 4, 5, 6, 7, 8, 9])
jldsave("logs/wcsp_il_random_seeds.jld2"; results=il_results)

train_idx = partialsortperm(train_rews, rev=false, 1:4)
val_idx = partialsortperm(val_rews, rev=false, 1:4)
mean_rews = [(train_rews[i] + val_rews[i]) / 2 for i in 1:length(train_rews)]
mean_idx = partialsortperm(mean_rews, rev=false, 1:4)
train_rews[train_idx]
val_rews[val_idx]
all_rews[2]
model_IL = all_rews[2].model;
train_IL = all_rews[2].train_rew
val_IL = all_rews[2].val_rew

# nb_epochs = 200, batch_size = 60, no_samples = 80, sigma_values = [0.05, 0.02], lr_values = [1e-3, 5e-4], use_critic = false, critic_steps = 0
# model 2: sigma = [0.1, 0.05], lr = [0.002, 0.001], temp = [0.1, 0.01]
fig = plot(train_IL, label="train history"; marker=:o)
plot!(fig, val_IL, label="val history"; marker=:o)
fig = plot(losses_IL, label="loss history"; marker=:o)
IL_final_train_rew = [cost(true_maximizer(model_IL(x)); c_true=kwargs.wg.weights) for (x, y, kwargs) in train_dataset];
IL_final_test_rew = [cost(true_maximizer(model_IL(x)); c_true=kwargs.wg.weights) for (x, y, kwargs) in test_dataset];
jldsave("logs/wcsp_il_best_model.jld2"; model=model_IL, train_rew=train_IL, val_rew=val_IL, train_final=IL_final_train_rew, test_final=IL_final_test_rew)

## PPO solution
Random.seed!(0);
critic_PPO = critic_warcraft_embedding();
model_PPO = new_warcraft_embedding();
model_PPO, train_PPO, val_PPO, losses_PPO = PPO_training(model_PPO, critic_PPO, train_dataset, val_dataset;
nb_epochs = 200, batch_size = 20, clip = 0.2, sigma_values = [0.1, 0.05], lr_values = [5e-4, 1e-4], use_critic = false, critic_steps = 0
)
PPO_train = mean([cost(true_maximizer(model_PPO(x)); c_true=kwargs.wg.weights) for (x, y, kwargs) in train_dataset]) # 33.8
ppo_runtime = @timed PPO_training(model_PPO, critic_PPO, train_dataset, val_dataset; nb_epochs = 200, batch_size = 20, clip = 0.2, sigma_values = [0.1, 0.05], lr_values = [5e-4, 1e-4], use_critic = false, critic_steps = 0)
ppo_runtime.time
# x, y, kwargs = train_dataset[1]
# model_PPO(x)

function PPO_runs(sigma_steps, lr_steps, seeds; soft=true)
    final_train = []
    final_val = []
    all_rews = []
    for i in sigma_steps
        for j in lr_steps
            for s in seeds
                Random.seed!(s);
                model_PPO = new_warcraft_embedding();
                model_PPO, train_PPO, val_PPO, losses_PPO = PPO_training(model_PPO, critic_PPO, train_dataset, val_dataset;
                nb_epochs = 200, batch_size = 20, clip = 0.2, sigma_values = i, lr_values = j, use_critic = false, critic_steps = 0
                )
                final_tr = [cost(true_maximizer(model_PPO(x)); c_true=kwargs.wg.weights) for (x, y, kwargs) in train_dataset]
                final_te = [cost(true_maximizer(model_PPO(x)); c_true=kwargs.wg.weights) for (x, y, kwargs) in test_dataset]
                push!(all_rews, (sigma=i, lr=j, seed=s, model=deepcopy(model_PPO), train_rew=train_PPO, val_rew=val_PPO, train_final=final_tr, test_final=final_te))
                push!(final_train, train_PPO[end])
                push!(final_val, val_PPO[end])
            end
        end
    end
    return final_train, final_val, all_rews
end

train_rews, val_rews, ppo_results = PPO_runs([[0.1, 0.05]], [[5e-4, 1e-4]], [1, 2, 3, 4, 5, 6, 7, 8, 9])
jldsave("logs/wcsp_ppo_random_seeds.jld2"; results=ppo_results)

# nb_epochs = 200, batch_size = 20, clip = 0.2, sigma_values = [0.1, 0.05], lr_values = [5e-4, 1e-4], use_critic = false, critic_steps = 0
fig = plot(train_PPO, label="train history"; marker=:o)
plot!(fig, val_PPO, label="val history"; marker=:o)
fig = plot(losses_PPO, label="loss history"; marker=:o)
PPO_final_train_rew = [cost(true_maximizer(model_PPO(x)); c_true=kwargs.wg.weights) for (x, y, kwargs) in train_dataset];
PPO_final_test_rew = [cost(true_maximizer(model_PPO(x)); c_true=kwargs.wg.weights) for (x, y, kwargs) in test_dataset];
jldsave("logs/wcsp_ppo_best_model.jld2"; model=model_PPO, train_rew=train_PPO, val_rew=val_PPO, train_final=PPO_final_train_rew, test_final=PPO_final_test_rew)

## Model tests
anticipative_test = mean([cost(y; c_true=kwargs.wg.weights) for (x, y, kwargs) in test_dataset]) # 29.75
greedy_test = mean([cost(greedy_path; c_true=kwargs.wg.weights) for (x, y, kwargs) in test_dataset]) # 43.53
random_test = mean([cost(true_maximizer(model_random(x)); c_true=kwargs.wg.weights) for (x, y, kwargs) in test_dataset]) # 43.53
SL_test = mean([cost(true_maximizer(model_SL(x)); c_true=kwargs.wg.weights) for (x, y, kwargs) in test_dataset]) # 30.83
IL_testrew = mean([cost(true_maximizer(model_IL(x)); c_true=kwargs.wg.weights) for (x, y, kwargs) in test_dataset]) # 30.28
PPO_test = mean([cost(true_maximizer(model_PPO(x)); c_true=kwargs.wg.weights) for (x, y, kwargs) in test_dataset]) # 32.4

SL_old = mean([cost(true_maximizer(encoder(x)); c_true=kwargs.wg.weights) for (x, y, kwargs) in test_dataset]) # 30.9
RM_old = mean([cost(true_maximizer(encoder(x)); c_true=kwargs.wg.weights) for (x, y, kwargs) in test_dataset]) # 33.56

greedy_train_rew = [cost(greedy_path; c_true=kwargs.wg.weights) for (x, y, kwargs) in train_dataset]
greedy_test_rew = [cost(greedy_path; c_true=kwargs.wg.weights) for (x, y, kwargs) in test_dataset]
opt_train_rew = [cost(y; c_true=kwargs.wg.weights) for (x, y, kwargs) in train_dataset]
opt_test_rew = [cost(y; c_true=kwargs.wg.weights) for (x, y, kwargs) in test_dataset]
jldsave("logs/wcsp_baselines.jld2"; greedy_train=greedy_train_rew, optimal_train=opt_train_rew, greedy_test=greedy_test_rew, optimal_test=opt_test_rew)