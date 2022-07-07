"""
    apply_learning_pipeline!(
        pipeline,
        pipeline_loss;
        true_maximizer,
        data_train,
        data_test,
        error_function,
        cost,
        epochs,
        verbose,
        setting_name="???",
)

Apply the learning pipeline `pipeline` based on the loss function `pipeline_loss`.
"""
function apply_learning_pipeline!(
    pipeline,
    pipeline_loss;
    true_maximizer,
    data_train,
    data_test,
    error_function,
    cost,
    epochs,
    verbose,
    setting_name="???",
)
    (; encoder, maximizer, loss) = pipeline
    @info "Testing $setting_name" maximizer loss

    ## Optimization
    opt = ADAM(0.001)
    perf_storage = init_perf()
    prog = Progress(epochs; enabled=verbose)

    for _ in 1:epochs
        next!(prog)
        update_perf!(
            perf_storage;
            data_train=data_train,
            data_test=data_test,
            encoder=encoder,
            true_maximizer=true_maximizer,
            pipeline_loss=pipeline_loss,
            error_function=error_function,
            cost=cost,
        )
        Flux.train!(pipeline_loss, Flux.params(encoder), zip(data_train...), opt)
    end

    ## Evaluation
    if verbose
        plts = plot_perf(perf_storage)
        for plt in plts
            println(plt)
        end
    end
    return
end
