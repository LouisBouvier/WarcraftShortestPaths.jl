"""
    save_metrics(;path::String,
                    losses::Matrix{Float64},
                    gaps::Matrix{Float64},
)

Save train and test losses and gaps tracked during training.
"""
function save_metrics(;
    path::String,
    losses::Matrix{Float64},
    gaps::Matrix{Float64},
)
    CSV.write(
        path *
        "losses.csv",
        Tables.table(losses),
        header = ["train", "test"]
    )
    CSV.write(
        path *
        "gaps.csv",
        Tables.table(gaps),
        header = ["train", "test"]
    )
    return
end