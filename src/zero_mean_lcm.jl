using LinearAlgebra
using Distributions
using Random
using Plots
using Optim

# David's notes: https://www.cs.cornell.edu/courses/cs6241/2025sp/lec/2025-03-11.html
struct LCM 
    X::Matrix{Float64}
    y::Vector{Float64}
    kernel::Function
    L::Matrix{Float64}
    lengthscale::Float64
    variance::Float64
    B::Matrix{Float64}

    function LCM(X::Matrix{Float64}, y::Vector{Float64}, kernel::Function,
    lengthscale::Float64, variance::Float64, B::Matrix{Float64})
        K = compute_kernel_matrix(X, X, kernel, lengthscale, variance, B)
        L = cholesky(K+1e-6*I(size(K, 1))).L
        return new(X, y, kernel, L, lengthscale, variance, B)
    end

    function LCM(X::Matrix{Float64}, y::Vector{Float64}, L::Matrix{Float64},
                kernel::Function, lengthscale::Float64, variance::Float64, B::Matrix{Float64})
        return new(X, y, kernel, L, lengthscale, variance, B)
    end
end

function compute_kernel_matrix(X1::Matrix{Float64}, X2::Matrix{Float64},
                              kernel::Function, lengthscale::Float64, variance::Float64,
                              B::Matrix{Float64})
    n1 = size(X1, 2)
    n2 = size(X2, 2)
    p = size(B, 1)  # Number of outputs
    K = zeros(n1 * p, n2 * p)
    K_base = zeros(n1, n2)
    for i in 1:n1
        for j in 1:n2
            K_base[i, j] = kernel(X1[:, i], X2[:, j], lengthscale, variance)
        end
    end
    for i in 1:p
        for j in 1:p
            K[(i-1)*n1+1:i*n1, (j-1)*n2+1:j*n2] = B[i, j] * K_base
        end
    end
    
    return K
end

function rbf_kernel(x1::Vector{Float64}, x2::Vector{Float64}, lengthscale, variance)
    return variance*exp(-0.5*sum((x1 .- x2).^2)/(lengthscale^2))
end

function predict(lcm::LCM, x_new::Matrix{Float64})
    K_inv = lcm.L' \ (lcm.L \ lcm.y)
    
    # Compute kernel matrices with the B parameter
    K21 = compute_kernel_matrix(x_new, lcm.X, lcm.kernel, lcm.lengthscale, lcm.variance,
                                lcm.B)
    K22 = compute_kernel_matrix(x_new, x_new, lcm.kernel, lcm.lengthscale, lcm.variance,
                                lcm.B)
    
    # Calculate mean and variance
    mu = K21 * K_inv
    v = lcm.L \ K21'
    σ2 = K22 - v' * v
    
    return mu, diag(σ2), K_inv
end

function EI(mean::Float64, std::Float64, f_best::Float64)
    z = (f_best - mean) / std
    cdf_z = cdf(Normal(), z)
    pdf_z = pdf(Normal(), z)
    
    return (f_best - mean)*cdf_z + std*pdf_z
end

function acquire_next_point(lcm::LCM, bounds::Matrix{Float64}, f_best::Float64;
                            n_samples::Int=1000)
    # Generate random samples within the bounds
    dimensions = size(bounds, 1)
    X_samples = zeros(dimensions, n_samples)
    for d in 1:dimensions
        X_samples[d, :] = bounds[1, d] .+ (bounds[2, d] - bounds[1, d])*rand(n_samples)
    end
    
    # Predict with the GP
    means, vars, _ = predict(lcm, X_samples)
    stds = sqrt.(vars)
    
    # Calculate EI for each sample
    # TODO: take every other mean to only use one of the functions in the multi-output.
    ei_values = [EI(means[i], stds[i], f_best) for i in 1:n_samples]
    
    # Find the point with maximum EI
    best_idx = argmax(ei_values)
    return X_samples[:, best_idx], ei_values[best_idx]
end

function update!(lcm::LCM, x::Vector{Float64}, y::Vector{Float64})
    X_new = [lcm.X x]
    y_new = [lcm.y; y]
    
    # Track how many data points we have after this update
    n_points = size(X_new, 2)
    
    if n_points % 10 == 0
        # Optimize hyperparameters
        lengthscale_new, variance_new, B_new = optimize_hyperparameters(X_new, y_new,
        lcm.kernel, lcm.lengthscale, lcm.variance)

        K = compute_kernel_matrix(X_new, X_new, lcm.kernel, lengthscale_new, variance_new, B_new)
        L_new = cholesky(K + 1e-6*I(size(K,1))).L
        L_new = Matrix(L_new)
    else
        # left-looking Cholesky update
        K12 = compute_kernel_matrix(lcm.X, reshape(x, :, 1), lcm.kernel, lcm.lengthscale,
        lcm.variance, lcm.B)
        K22 = compute_kernel_matrix(reshape(x, :, 1), reshape(x, :, 1), lcm.kernel,
        lcm.lengthscale, lcm.variance, lcm.B)
        
        L12 = lcm.L \ K12
        L22 = sqrt(K22 - L12' * L12) .+ 1e-6*I(size(K22, 1))
        L_new = [lcm.L zeros(Float64, size(lcm.L, 1), 1); 
                 reshape(L12', 1, :) L22]
        
        # Use existing hyperparameters (not optimizing every iteration)
        lengthscale_new = lcm.lengthscale
        variance_new = lcm.variance
        B_new = lcm.B
    end
    return LCM(X_new, y_new, L_new, lcm.kernel, lengthscale_new, variance_new, B_new)
end

function log_marginal_likelihood(L::Matrix{Float64}, α::Vector{Float64},
                                 y::Vector{Float64}, params::Vector{Float64})
    lengthscale = params[1]
    variance = params[2]
    B = params[3]
    
    n = length(y)
    return -0.5 * dot(y, α) - sum(log.(diag(L))) - 0.5 * n * log(2π)
end

# Optimize hyperparameters using BFGS
function optimize_hyperparameters(lcm::LCM, L::Matrix{Float64}, α::Vector{Float64},
                                  y::Vector{Float64}; method=LBFGS())
    # Initial parameter values
    initial_params = [lcm.lengthscale, lcm.variance, lcm.B]

    # Set lower bounds to ensure positive values
    lower_bounds = [1e-6, 1e-6]
    upper_bounds = [10.0, 10.0]
    
    # Define the negative log marginal likelihood function to minimize
    function objective(params)
        return -log_marginal_likelihood(L, α, y, initial_params)
    end
    
    # Run optimization
    result = Optim.optimize(objective, lower_bounds, upper_bounds, initial_params, Fminbox(method))
    
    # Get optimized parameters
    opt_params = Optim.minimizer(result)
    opt_lengthscale = opt_params[1]
    opt_variance = opt_params[2]
    opt_B = opt_params[3]
    
    # Create a new GP with optimized hyperparameters
    return opt_lengthscale, opt_variance, opt_B
end

function BO_loop(f::Function, bounds::Matrix{Float64}, n_iterations::Int;
                n_init::Int=5)
    
    dimensions = size(bounds, 1)
    
    lengthscale = 0.5
    variance = 1.2
    
    # Ensure B matrix is positive definite
    L_B = [1.0 0.0; 0.5 0.866]
    B = L_B * L_B'
    
    num_outputs = size(B, 1)  # Number of outputs
    
    # Initialize with random points
    X_init = zeros(dimensions, n_init)
    for d in 1:dimensions
        X_init[d, :] = bounds[1, d] .+ (bounds[2, d] - bounds[1, d]) * rand(n_init)
    end
    
    # Evaluate function for each input and flatten the outputs
    y_all_outputs = [f(X_init[:, i]) for i in 1:n_init]
    y_init = vcat([y_all_outputs[i][j] for i in 1:n_init for j in 1:num_outputs]...)
    
    # Create and fit GP
    lcm = LCM(X_init, y_init, rbf_kernel, lengthscale, variance, B)

    # Keep track of all evaluated points
    X_all = copy(X_init)
    y_all = copy(y_init)
    
    # Extract first output values (assuming we optimize the first output)
    # For a vector [y1_1, y2_1, y1_2, y2_2, ...], we want [y1_1, y1_2, ...]
    first_output_indices = 1:num_outputs:length(y_init)
    first_output_values = y_init[first_output_indices]
    
    # Find the index of the best value among the first output values
    best_output_idx = argmin(first_output_values)
    
    # Convert to the original input index
    # Since each input point has num_outputs outputs, we need to map back
    best_input_idx = best_output_idx
    
    f_best = first_output_values[best_output_idx]
    x_best = X_init[:, best_input_idx]
    
    # Optimization history
    history = [(x_best, f_best)]
    
    # Main optimization loop
    for i in 1:n_iterations
        # Select next point
        next_x, ei = acquire_next_point(lcm, bounds, f_best)
        
        # Evaluate function
        next_y_vector = f(next_x)
        
        # Get first output for optimization criterion
        next_y = next_y_vector[1]
        
        # Flatten to a single vector for the LCM model
        next_y_flat = vcat(next_y_vector...)
        
        # Update GP
        lcm = update!(lcm, next_x, next_y_flat)
        
        # Update records
        X_all = [X_all next_x]
        y_all = [y_all; next_y_flat]
        
        # Update best observation (using first output)
        if next_y < f_best
            f_best = next_y
            x_best = next_x
            push!(history, (x_best, f_best))
        end

        println("Iteration $i: x = $next_x, f(x) = $next_y, Best = $f_best")
    end
    
    lengthscale_final = lcm.lengthscale
    variance_final = lcm.variance
    println("Final lengthscale $lengthscale_final, final variance $variance_final")
    
    return x_best, f_best, history, lcm 
end

# Example usage script
function main()
    # Define test function (2D Branin function with 2 outputs)
    function branin(x)
        x1 = x[1]
        x2 = x[2]
        
        # First output
        term1 = (x2 - 5.1/(4*π^2) * x1^2 + 5/π * x1 - 6)^2
        term2 = 10 * (1 - 1/(8π)) * cos(x1)
        y1 = term1 + term2 + 10
        
        # Second output (correlated but different)
        y2 = 0.5 * term1 + 1.5 * term2 + 5
        
        # Return both outputs as a vector
        return [y1, y2]  
    end
    
    # Define bounds
    bounds = [-5.0 0.0; 10.0 15.0]  # [min_x1 min_x2; max_x1 max_x2]
    
    # Run Bayesian optimization
    x_best, f_best, history, lcm = BO_loop(branin, bounds, 30, n_init=10)
    
    println("Best solution found: x = $x_best, f(x) = $f_best")
    
    # Visualize results
    x1_range = range(bounds[1,1], bounds[2,1], length=100)
    x2_range = range(bounds[1,2], bounds[2,2], length=100)
    
    z = [branin([x1, x2])[1] for x1 in x1_range, x2 in x2_range]
    
    p = surface(x1_range, x2_range, z, alpha=0.7,
                title="Objective Function with Best Point")
    scatter!([x_best[1]], [x_best[2]], [f_best], markersize=5, 
             color=:red, label="Best Point")
    
    display(p)
    
    # Plot convergence
    convergence = [h[2] for h in history]
    p2 = plot(convergence, marker=:circle, 
              xlabel="Iteration", ylabel="Best Value Found",
              title="Convergence Plot", label="Best Value")
    display(p2)
end

# Run the example
main()