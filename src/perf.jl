## Performance metrics
"""
    function init_perf()

Initialize the performance metrics followed during training.

The six metrics we consider are:
 - The total train and test losses.
 - The train and test average errors of the predicted solutions compared with the true solutions.
 - The train and test cost gaps with respect to the true solutions.
"""

function init_perf()
    perf_storage = (
        train_losses=Float64[],
        test_losses=Float64[],
        train_errors=Float64[],
        test_errors=Float64[],
        train_cost_gaps=Float64[],
        test_cost_gaps=Float64[],
    )
    return perf_storage
end

"""
    update_perf!(
        perf_storage::NamedTuple;
        data_train,
        data_test,
        encoder,
        true_maximizer,
        pipeline_loss,
        error_function,
        cost,
)

Update the performance metrics during training.

The six metrics we consider are:
 - The total train and test losses.
 - The train and test average errors of the predicted solutions compared with the true solutions.
 - The train and test cost gaps with respect to the true solutions.
"""
function update_perf!(
    perf_storage::NamedTuple;
    data_train,
    data_test,
    encoder,
    true_maximizer,
    pipeline_loss,
    error_function,
    cost,
)
    (;
        train_losses,
        test_losses,
        train_errors,
        test_errors,
        train_cost_gaps,
        test_cost_gaps,
    ) = perf_storage

    (X_train, thetas_train, Y_train) = data_train
    (X_test, thetas_test, Y_test) = data_test

    train_loss = sum(pipeline_loss(x, θ, y) for (x, θ, y) in zip(data_train...))
    test_loss = sum(pipeline_loss(x, θ, y) for (x, θ, y) in zip(data_test...))

    Y_train_pred = generate_predictions(encoder, true_maximizer, X_train)
    Y_test_pred = generate_predictions(encoder, true_maximizer, X_test)

    train_error = mean(
        error_function(y_pred, y) for (y, y_pred) in zip(Y_train, Y_train_pred)
    )
    test_error = mean(error_function(y_pred, y) for (y, y_pred) in zip(Y_test, Y_test_pred))

    train_cost = [cost(y, θ) for (y, θ) in zip(Y_train_pred, thetas_train)]
    train_cost_opt = [cost(y, θ) for (y, θ) in zip(Y_train, thetas_train)]
    test_cost = [cost(y, θ) for (y, θ) in zip(Y_test_pred, thetas_test)]
    test_cost_opt = [cost(y, θ) for (y, θ) in zip(Y_test, thetas_test)]

    train_cost_gap = mean(
        (c - c_opt) / abs(c_opt) for (c, c_opt) in zip(train_cost, train_cost_opt)
    )
    test_cost_gap = mean(
        (c - c_opt) / abs(c_opt) for (c, c_opt) in zip(test_cost, test_cost_opt)
    )


    push!(train_losses, train_loss)
    push!(test_losses, test_loss)
    push!(train_errors, train_error)
    push!(test_errors, test_error)
    push!(train_cost_gaps, train_cost_gap)
    push!(test_cost_gaps, test_cost_gap)
    return nothing
end

"""
    plot_perf(perf_storage::NamedTuple)

Create and display unicode plots of the performance metrics stored in `perf_storage`.
"""

function plot_perf(perf_storage::NamedTuple)
    (;
        train_losses,
        test_losses,
        train_errors,
        test_errors,
        train_cost_gaps,
        test_cost_gaps,
    ) = perf_storage
    plts = []

    if any(!isnan, train_losses)
        plt = lineplot(train_losses; xlabel="Epoch", title="Train loss")
        push!(plts, plt)
    end

    if any(!isnan, test_losses)
        plt = lineplot(test_losses; xlabel="Epoch", title="Test loss")
        push!(plts, plt)
    end

    if any(!isnan, train_errors)
        plt = lineplot(
            train_errors;
            xlabel="Epoch",
            title="Train error",
            # ylim=(0, maximum(train_errors)),
        )
        push!(plts, plt)
    end

    if any(!isnan, test_errors)
        plt = lineplot(
            test_errors;
            xlabel="Epoch",
            title="Test error",
            # ylim=(0, maximum(test_errors))
        )
        push!(plts, plt)
    end

    if any(!isnan, train_cost_gaps)
        plt = lineplot(
            train_cost_gaps;
            xlabel="Epoch",
            title="Train cost gap",
            # ylim=(0, maximum(train_cost_gaps)),
        )
        push!(plts, plt)
    end

    if any(!isnan, train_cost_gaps)
        plt = lineplot(
            test_cost_gaps;
            xlabel="Epoch",
            title="Test cost gap",
            # ylim=(0, maximum(test_cost_gaps)),
        )
        push!(plts, plt)
    end

    return plts
end
