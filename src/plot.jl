# plots.jl
using Plots

"""
    plot_gp_results(X_train, y_train, X_test, y_pred, y_std; 
                    title="Gaussian Process Regression", 
                    true_func=sin)

Plot training data, predictions, and confidence intervals for a Gaussian process model.
Optionally overlays the true function if provided.
"""
function plot_gp_results(X_train, y_train, X_test, y_pred, y_std; 
                        title="Gaussian Process Regression", 
                        true_func=nothing)
    p = plot(title=title, xlabel="x", ylabel="y", 
            legend=:topleft, size=(800, 600))
    
    # Sort test inputs for smooth plotting
    sorted_indices = sortperm(X_test)
    X_test_sorted = X_test[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    y_std_sorted = y_std[sorted_indices]
    
    # Plot confidence intervals (95%)
    confidence_band = 1.96 * y_std_sorted
    plot!(X_test_sorted, y_pred_sorted, ribbon=confidence_band, 
          fillalpha=0.3, color=:blue, label="Predictions with 95% CI")
    
    # Plot training data
    scatter!(X_train, y_train, color=:red, markersize=5, label="Training data")
    
    # Plot predictions
    plot!(X_test_sorted, y_pred_sorted,
          color=:blue,
          linewidth=2,
          label="Predictions")
    
    # Plot true function if provided
    if true_func !== nothing
        x_dense = range(minimum(X_test), maximum(X_test), length=200)
        plot!(x_dense, true_func.(x_dense), color=:green, linestyle=:dash, 
              linewidth=2, label="True function")
    end
    
    return p
end

"""
    save_gp_plot(p, filename="gp_plot.png")

Save the Gaussian process plot to a file.
"""
function save_gp_plot(p, filename="gp_plot.png")
    savefig(p, filename)
    println("Plot saved as $filename")
end