# main.jl
include("base.jl")
include("inference.jl")
include("plot.jl")

function main()
    println("Running Gaussian Process Inference")
    
    # Generate or load data
    X, y = generate_synthetic_data(30, 0.1)
    
    # Split into training and testing sets
    X_train, y_train, X_test, y_test = split_data(X, y, 0.8)
    
    # Create a denser test set for smooth predictions
    X_dense = range(0, 2π, length=100)
    
    # Perform GP inference
    l = 0.5  # length scale hyperparameter
    sigma = 1.0  # signal variance hyperparameter
    noise_var = 0.1  # noise variance
    
    println("Training GP model...")
    y_pred, y_std = predict_with_uncertainty(X_train, y_train, X_dense, rbf_kernel, noise_var, l, sigma)
    
    # Plot results
    println("Creating plot...")
    p = plot_gp_results(X_train, y_train, X_dense, y_pred, y_std, true_func=sin)
    
    # Save plot
    save_gp_plot(p, "gp_inference_result.png")
    
    println("Done! Check 'gp_inference_result.png' for the plot.")
end

# Run the main function
main()