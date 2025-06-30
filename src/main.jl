using Plots
include("zero_mean_gp.jl")

function branin(x)
    a = 1.0
    b = 5.1 / (4π^2)
    c = 5.0 / π
    r = 6.0
    s = 10.0
    t = 1.0 / (8π)

    return a * (x[2] - b * x[1]^2 + c * x[1] - r)^2 + s * (1 - t) * cos(x[1]) + s
end

function plot_bo_results(gp, X_all, x_best)
    x1_range = range(-5, 10, length=100)
    x2_range = range(0, 15, length=100)

    z = zeros(length(x2_range), length(x1_range))
    for (i, x1) in enumerate(x1_range)
        for (j, x2) in enumerate(x2_range)
            z[j, i] = branin([x1, x2])
        end
    end

    # Create GP prediction grid
    X_pred = zeros(2, length(x1_range) * length(x2_range))
    idx = 1
    for (i, x1) in enumerate(x1_range)
        for (j, x2) in enumerate(x2_range)
            X_pred[:, idx] = [x1, x2]
            idx += 1
        end
    end

    means, vars = predict(gp, X_pred)
    mean_grid = reshape(means, (length(x2_range), length(x1_range)))

    # Create single plot with true function contours and GP fit
    p = contour(x1_range, x2_range, z, levels=20, color=:viridis, alpha=0.7,
              xlabel="x₁", ylabel="x₂", title="Branin Function with GP Fit")

    # Add sampled points
    scatter!(p, [X_all[1, i] for i in 1:size(X_all, 2)], 
              [X_all[2, i] for i in 1:size(X_all, 2)], 
              color=:white, markersize=5, label="Samples")

    # Add best point
    scatter!(p, [x_best[1]], [x_best[2]], color=:red, markersize=8, 
            markershape=:star5, label="Best Solution")

    # Add GP contours
    contour!(p, x1_range, x2_range, mean_grid, levels=10, 
            linestyle=:dash, linewidth=2, color=:orange, alpha=0.8,
            label="GP Mean")

    return p
end

#bounds = [[-5.0, 10.0] [0.0, 15.0]]
#x_best, f_best, history, gp = BO_loop(branin, bounds, 50, n_init=20)
#p = plot_bo_results(gp, gp.X, x_best)
#savefig(p, "zero_mean_gp.svg")
#println("Plot saved as 'zero_mean_gp.svg'")