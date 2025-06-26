using LinearAlgebra
using Distributions
using Random
using Plots
using Optim

# David's notes: https://www.cs.cornell.edu/courses/cs6241/2025sp/lec/2025-03-11.html

function chebyshev_polynomial(x, n)
    # Map input to [-1, 1] where Chebyshev polynomials are defined
    x_mapped = 2*(x .- minimum(x))/(maximum(x) - minimum(x)) .- 1
    
    if n == 0
        T1 = ones(size(x))
    elseif n == 1
        T1 = x_mapped
    else
        T0 = ones(size(x))
        T1 = x_mapped
        for i in 2:n
            T2 = 2 .* x_mapped .* T1 .- T0
            T0 = T1
            T1 = T2
        end
    end
    return T1
end

# Create a structure for the Chebyshev mean function
struct ChebyshevMean
    degree::Int
    coefficients::Vector{Float64}
    
    # Constructor with default coefficients (all zeros)
    function ChebyshevMean(degree::Int)
        coefficients = zeros(degree + 1)
        return new(degree, coefficients)
    end
    
    # Constructor with provided coefficients
    function ChebyshevMean(degree::Int, coefficients::Vector{Float64})
        if length(coefficients) != degree + 1
            error("Coefficients length must be degree + 1")
        end
        return new(degree, coefficients)
    end
end

# Function to evaluate the mean at given points
function evaluate_mean(mean_func::ChebyshevMean, X::Matrix{Float64})
    n = size(X, 2)
    result = zeros(n)
    
    for i in 0:mean_func.degree
        for d in 1:size(X, 1)  # For each dimension
            chebyshev_values = chebyshev_polynomial(X[d, :], i)
            result .+= mean_func.coefficients[i+1] * chebyshev_values
        end
    end
    
    return result
end

struct GP
    X::Matrix{Float64}
    y::Vector{Float64}
    kernel::Function
    mean_func::ChebyshevMean
    L::Matrix{Float64}
    lengthscale::Float64
    variance::Float64

    # Constructor with mean function
    function GP(X::Matrix{Float64}, y::Vector{Float64}, kernel::Function,
                mean_func::ChebyshevMean, lengthscale::Float64, variance::Float64)
        K = compute_kernel_matrix(X, X, kernel, lengthscale, variance)
        K += 1e-6 * I
        L = cholesky(K).L
        return new(X, y, kernel, mean_func, L, lengthscale, variance)
    end

    # Constructor with pre-computed L
    function GP(X::Matrix{Float64}, y::Vector{Float64}, L::Matrix{Float64},
                kernel::Function, mean_func::ChebyshevMean, lengthscale::Float64,
                variance::Float64)
        return new(X, y, kernel, mean_func, L, lengthscale, variance)
    end
end

function compute_kernel_matrix(X1::Matrix{Float64}, X2::Matrix{Float64},
                               kernel::Function, lengthscale::Float64, variance::Float64)
    n1 = size(X1, 2)
    n2 = size(X2, 2)
    K = zeros(n1, n2)

    for i in 1:n1
        for j in 1:n2
            K[i, j] = kernel(X1[:, i], X2[:, j], lengthscale, variance)
        end
    end

    return K
end

function rbf_kernel(x1::Vector{Float64}, x2::Vector{Float64}, lengthscale, variance)
    return variance*exp(-0.5*sum((x1 .- x2).^2)/(lengthscale^2))
end

function predict(gp::GP, x_new::Matrix{Float64})
    μ_train = evaluate_mean(gp.mean_func, gp.X)
    centered_y = gp.y - μ_train
    K_inv = gp.L' \ (gp.L \ centered_y)
    K21 = compute_kernel_matrix(x_new, gp.X, gp.kernel, gp.lengthscale, gp.variance)
    K22 = compute_kernel_matrix(x_new, x_new, gp.kernel, gp.lengthscale, gp.variance)
    K22 += 1e-6 * I
    μ = K21 * K_inv
    prior = evaluate_mean(gp.mean_func, x_new)
    μ = prior + μ

    v = gp.L \ K21'
    σ2 = K22 - v' * v

    return μ, diag(σ2), K_inv
end

function EI(mean::Float64, std::Float64, f_best::Float64)
    z = (f_best - mean) / std
    cdf_z = cdf(Normal(), z)
    pdf_z = pdf(Normal(), z)
    
    return (f_best - mean)*cdf_z + std*pdf_z
end

function acquire_next_point(gp::GP, bounds::Matrix{Float64}, f_best::Float64;
                            n_samples::Int=1000)
    dimensions = size(bounds, 1)
    
    # Generate random samples within the bounds
    X_samples = zeros(dimensions, n_samples)
    for d in 1:dimensions
        X_samples[d, :] = bounds[1, d] .+ (bounds[2, d] - bounds[1, d])*rand(n_samples)
    end
    
    # Predict with the GP
    means, vars, _ = predict(gp, X_samples)
    stds = sqrt.(vars)
    
    # Calculate EI for each sample
    ei_values = [EI(means[i], stds[i], f_best) for i in 1:n_samples]
    
    # Find the point with maximum EI
    best_idx = argmax(ei_values)
    return X_samples[:, best_idx], ei_values[best_idx]
end

function log_marginal_likelihood(X::Matrix{Float64}, y::Vector{Float64}, 
                               L::Matrix{Float64}, mean_func::ChebyshevMean,
                               params::Vector{Float64})
    # Extract kernel parameters
    lengthscale = params[1]
    variance = params[2]
    
    mean_coef = params[3:end]
    new_mean_func = ChebyshevMean(mean_func.degree, mean_coef)
    mean_train = evaluate_mean(new_mean_func, X)
    
    centered_y = y - mean_train
    
    α = L' \ (L \ centered_y)
    
    n = length(y)
    return -0.5 * dot(centered_y, α) - sum(log.(diag(L))) - 0.5 * n * log(2π)
end

# Modify the optimize_hyperparameters function
function optimize_hyperparameters(gp::GP, X::Matrix{Float64}, L::Matrix{Float64},
                                  α::Vector{Float64}, y::Vector{Float64}; method=BFGS())
    # Initial parameter values: [lengthscale, variance, mean_coefficients...]
    mean_degree = gp.mean_func.degree
    initial_params = [gp.lengthscale, gp.variance, gp.mean_func.coefficients...]
    
    # Set bounds
    lower_bounds = [1e-6, 1e-6, fill(-10.0, mean_degree+1)...]
    upper_bounds = [10.0, 10.0, fill(10.0, mean_degree+1)...]
    
    # Define the negative log marginal likelihood function to minimize
    function objective(params)
        return -log_marginal_likelihood(X, y, L, gp.mean_func, params)
    end
    
    # Run optimization
    result = Optim.optimize(objective, lower_bounds, upper_bounds, initial_params,
                            Fminbox(method))
    
    # Get optimized parameters
    opt_params = Optim.minimizer(result)
    opt_lengthscale = opt_params[1]
    opt_variance = opt_params[2]
    opt_mean_coef = opt_params[3:end]
    
    # Return optimized parameters
    return opt_lengthscale, opt_variance, opt_mean_coef
end

# Update the GP update function
function update!(gp::GP, x::Vector{Float64}, y::Float64)
    X_new = [gp.X x]
    y_new = [gp.y; y]
    
    # Update the Cholesky factor L
    K12 = compute_kernel_matrix(gp.X, reshape(x, :, 1), gp.kernel, gp.lengthscale, gp.variance)
    K22 = compute_kernel_matrix(reshape(x, :, 1), reshape(x, :, 1), gp.kernel, gp.lengthscale, gp.variance)
    K22 += 1e-6 * I
    L12 = gp.L \ K12
    L22 = K22 - L12' * L12
    L22 = sqrt(max(1e-10, L22[1, 1]))

    L_new = [gp.L zeros(Float64, size(gp.L, 1), 1); reshape(L12, 1, :) L22]
    
    # Calculate mean at training points and center observations
    mean_train = evaluate_mean(gp.mean_func, X_new)
    centered_y = y_new - mean_train
    
    # Calculate alpha for optimization
    α = L_new' \ (L_new \ centered_y)
    
    # Optimize hyperparameters
    lengthscale_new, variance_new, mean_coef_new = optimize_hyperparameters(gp, X_new,
    L_new, α, y_new)
    
    # Create new mean function with optimized coefficients
    mean_func_new = ChebyshevMean(gp.mean_func.degree, mean_coef_new)
    
    # Return updated GP
    return GP(X_new, y_new, L_new, gp.kernel, mean_func_new, lengthscale_new,
    variance_new)
end

function BO_loop(f::Function, bounds::Matrix{Float64}, n_iterations::Int;
                 n_init::Int=5, cheby_degree::Int=3) 
    dimensions = size(bounds, 1)
    
    # Initial kernel parameters
    lengthscale = 0.5
    variance = 1.2
    
    # Create initial mean function
    mean_func = ChebyshevMean(cheby_degree)
    
    # Initialize with random points
    X_init = zeros(dimensions, n_init)
    for d in 1:dimensions
        X_init[d, :] = bounds[1, d] .+ (bounds[2, d] - bounds[1, d]) * rand(n_init)
    end
    y_init = [f(X_init[:, i]) for i in 1:n_init]
    
    # Create and fit GP with mean function
    gp = GP(X_init, y_init, rbf_kernel, mean_func, lengthscale, variance)

    # Keep track of all evaluated points
    X_all = copy(X_init)
    y_all = copy(y_init)
    f_best = minimum(y_init)
    x_best = X_init[:, argmin(y_init)]
    
    # Optimization history
    history = [(x_best, f_best)]
    
    # Main optimization loop
    for i in 1:n_iterations
        # Select next point
        next_x, ei = acquire_next_point(gp, bounds, f_best)
        
        # Evaluate function
        next_y = f(next_x)
        
        # Update GP
        gp = update!(gp, next_x, next_y)
        
        # Update records
        X_all = [X_all next_x]
        y_all = [y_all; next_y]
        
        # Update best observation
        if next_y < f_best
            f_best = next_y
            x_best = next_x
            push!(history, (x_best, f_best))
        end

        println("Iteration $i: x = $next_x, f(x) = $next_y, Best = $f_best")
    end
    lengthscale_final = gp.lengthscale
    variance_final = gp.variance
    println("Final lengthscale $lengthscale_final, final variance $variance_final")
    
    return x_best, f_best, history, gp
end

using Plots

function plot_bo_results(gp, X_all, x_best)
    # Create grid for the Branin function
    x1_range = range(-5, 10, length=100)
    x2_range = range(0, 15, length=100)
    
    # Branin function for reference
    function branin(x)
        a = 1.0
        b = 5.1 / (4π^2)
        c = 5.0 / π
        r = 6.0
        s = 10.0
        t = 1.0 / (8π)
        
        return a * (x[2] - b * x[1]^2 + c * x[1] - r)^2 + s * (1 - t) * cos(x[1]) + s
    end
    
    # Compute function values on grid
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
    
     # Get prior mean on grid
    prior_mean = evaluate_mean(gp.mean_func, X_pred)
    prior_mean_grid = reshape(prior_mean, (length(x2_range), length(x1_range)))
    
    # Get posterior predictions
    means, vars = predict(gp, X_pred)
    mean_grid = reshape(means, (length(x2_range), length(x1_range)))
    
    # Plot multiple subplots
    p1 = contour(x1_range, x2_range, z, levels=20, color=:viridis, alpha=0.7, xlabel="x₁",
    ylabel="x₂", title="True Function")
              
    p2 = contour(x1_range, x2_range, prior_mean_grid, levels=20, color=:thermal,
    alpha=0.7, xlabel="x₁", ylabel="x₂", title="Prior Mean")
              
    p3 = contour(x1_range, x2_range, mean_grid, levels=20, color=:viridis, alpha=0.7,
    xlabel="x₁", ylabel="x₂", title="Posterior Mean")
    
    # Add sampled points and best point to posterior plot
    scatter!(p3, [X_all[1, i] for i in 1:size(X_all, 2)], 
              [X_all[2, i] for i in 1:size(X_all, 2)], 
              color=:white, markersize=5, label="Samples")
              
    scatter!(p3, [x_best[1]], [x_best[2]], color=:red, markersize=8, 
            markershape=:star5, label="Best Solution")
    
    # Combine plots
    p = plot(p1, p2, p3, layout=(1,3), size=(900, 300))
    
    return p
end

function run_and_save_plot()
    # Define bounds for Branin function
    bounds = [[-5.0, 10.0] [0.0, 15.0]]
    
    # Define Branin function
    function branin(x)
        a = 1.0
        b = 5.1 / (4π^2)
        c = 5.0 / π
        r = 6.0
        s = 10.0
        t = 1.0 / (8π)
        
        return a * (x[2] - b * x[1]^2 + c * x[1] - r)^2 + s * (1 - t) * cos(x[1]) + s
    end
    
    # Run Bayesian optimization
    x_best, f_best, history, gp = BO_loop(branin, bounds, 100, n_init=20)
    
    # Plot results
    p = plot_bo_results(gp, gp.X, x_best)
    
    # Save as PNG with high resolution
    savefig(p, "bo_optim_plot_nnz.png")
    
    println("Plot saved as 'bo_optim_plot_nnz.png'")
    
    return p
end

# Run the function to generate and save the plot
run_and_save_plot()
