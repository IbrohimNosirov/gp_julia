using LinearAlgebra
using Distributions
using Random
using Plots
using Optim

# David's notes: https://www.cs.cornell.edu/courses/cs6241/2025sp/lec/2025-03-11.html

struct GP
    X::Matrix{Float64}
    y::Vector{Float64}
    kernel::Function
    L::Matrix{Float64}
    lengthscale::Float64
    variance::Float64

    function GP(X::Matrix{Float64}, y::Vector{Float64}, kernel::Function,
    lengthscale::Float64, variance::Float64)
        K = compute_kernel_matrix(X, X, kernel, lengthscale, variance)
        L = cholesky(K).L
        return new(X, y, kernel, L, lengthscale, variance)
    end

    function GP(X::Matrix{Float64}, y::Vector{Float64}, L::Matrix{Float64},
                kernel::Function, lengthscale::Float64, variance::Float64)
        return new(X, y, kernel, L, lengthscale, variance)
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

function grad_kernel_matrix(gp::GP, x::Vector{Float64})
    n1 = size(gp.X)
    k = zeros(n1)
    for i in 1:n1
        k[i] = -2 * rbf_kernel(gp.X[:, i], x, gp.lengthscale, gp.variance)*(X[:, i] - x)
        k[i] /= gp.lengthscale^2
    end
    return k
end

function predict(gp::GP, x_new::Matrix{Float64})
    K_inv = gp.L' \ (gp.L \ gp.y)
    K21 = compute_kernel_matrix(x_new, gp.X, gp.kernel, gp.lengthscale, gp.variance)
    K22 = compute_kernel_matrix(x_new, x_new, gp.kernel, gp.lengthscale, gp.variance)
    mu = K21 * K_inv
    v = gp.L \ K21'
    sigma2 = K22 - v' * v

    return mu, diag(sigma2), K_inv
end

# need to compute derivative of this or either log EI.
function EI(gp::GP, x::Matrix{Float64}, f_best::Float64)
    mean, var = predict(gp, x)
    z = (f_best - mean) / std
    cdf_z = cdf(Normal(), z)
    pdf_z = pdf(Normal(), z)
    return (f_best - mean)*cdf_z + std*pdf_z
end

# HUGE DISCLAIMER: NEED TO TEST THOROUGHLY
function compute_grad_log_EI(gp::GP, x::Matrix{Float64}, f_best)
    mean, var = predict(gp, x)
    z = (f_best - mean) / std
    cdf_z = cdf(Normal(), z)
    pdf_z = pdf(Normal(), z)
    EI = (f_best - mean)*cdf + std*pdf
    grad_x_k, k = grad_kernel_matrix(gp, x)
    grad_mu = grad_x_k * L' \ (L \ y)
    grad_sigma = -2 * grad_x_k' * (L \ k)

    return (-cdf_z * grad_mu + pdf_z * grad_sigma)/EI
end

function backtracking_line_search(gp::GP, x_current::Matrix{Float64})
    alpha = 1
    rho = 0.9
    c = 0.9
    p = -grad_log_EI/norm(grad_log_EI)
    lhs, _ = predict(gp, x_k + alpha*p)
    rhs, _ = predict(gp, x_k)
    rhs += c*alpha*grad_log_EI * p
    while lhs > rhs
        lhs, _ = predict(gp, x_k + alpha*p)
        rhs, _ = predict(gp, x_k)
        rhs += c*alpha*grad_log_EI * p
        alpha = rho * alpha
    end
    return alpha
end

function acquire_next_point(x_current::Matrix{Float64})
    grad_log_EI = compute_grad_log_EI(x_current)
    step_size = backtracking_line_search(x_current)
    x_next = x_current - step_size*grad_log_EI
end

function update!(gp::GP, x::Vector{Float64}, y::Float64)
    X_new = [gp.X x]
    y_new = [gp.y; y]
    
    # Track how many data points we have after this update
    n_points = size(X_new, 2)
    
    if n_points % 5 == 0
        # Optimize hyperparameters
        lengthscale_new, variance_new = optimize_hyperparameters(X_new, y_new,
        gp.kernel, gp.lengthscale, gp.variance)

        K = compute_kernel_matrix(X_new, X_new, gp.kernel, lengthscale_new, variance_new)
        L_new = cholesky(K + 1e-10*I(size(K,1))).L
        L_new = Matrix(L_new)
    else
        # left-looking Cholesky update
        K12 = compute_kernel_matrix(gp.X, reshape(x, :, 1), gp.kernel, gp.lengthscale,
        gp.variance)
        K22 = compute_kernel_matrix(reshape(x, :, 1), reshape(x, :, 1), gp.kernel,
        gp.lengthscale, gp.variance)
        
        L12 = gp.L \ K12
        L22 = sqrt(K22 - L12' * L12) .+ 1e-6
        L_new = [gp.L zeros(Float64, size(gp.L, 1), 1); 
                    reshape(L12', 1, :) L22]
        
        # Use existing hyperparameters (not optimizing every iteration)
        lengthscale_new = gp.lengthscale
        variance_new = gp.variance
    end
    return GP(X_new, y_new, L_new, gp.kernel, lengthscale_new, variance_new)
end

function optimize_hyperparameters(X, y, kernel, lengthscale, variance; method=LBFGS())
    initial_params = [lengthscale, variance]
    lower_bounds = [1e-6, 1e-6]
    upper_bounds = [10.0, 10.0]
    
    function objective(params)
        return -log_marginal_likelihood(X, y, kernel, params)
    end
    
    result = Optim.optimize(objective, lower_bounds, upper_bounds, initial_params,
    Fminbox(method))
    
    # Get optimized parameters
    opt_params = Optim.minimizer(result)
    opt_lengthscale = opt_params[1]
    opt_variance = opt_params[2]
    
    return opt_lengthscale, opt_variance
end

function log_marginal_likelihood(X::Matrix{Float64}, y::Vector{Float64}, kernel::Function,
                                 params::Vector{Float64})
    lengthscale = params[1]
    variance = params[2]
    
    # Compute kernel matrix with current hyperparameters
    K = compute_kernel_matrix(X, X, kernel, lengthscale, variance)
    
    # Add small jitter for numerical stability
    n = size(K, 1)
    # Cholesky decomposition
    L = cholesky(K + 1e-6*I(size(K,1))).L
    
    # Compute alpha
    α = L' \ (L \ y)
    
    # Compute log marginal likelihood
    return -0.5 * dot(y, α) - sum(log.(diag(L))) - 0.5 * n * log(2π)
end

#TODO: make a function gradient of log-marginal likelihood

#TODO: model_update()

function BO_loop(f::Function, bounds::Matrix{Float64}, n_iterations::Int;
                 n_init::Int=5)
    dimensions = size(bounds, 1)
    lengthscale = 0.5
    variance = 1.2
    # Initialize with random points
    X_init = zeros(dimensions, n_init)
    for d in 1:dimensions
        X_init[d, :] = bounds[1, d] .+ (bounds[2, d] - bounds[1, d]) * rand(n_init)
    end
    y_init = [f(X_init[:, i]) for i in 1:n_init]
    
    # Create and fit GP
    gp = GP(X_init, y_init, rbf_kernel, lengthscale, variance)

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
        
        #TODO: every few iterations, model_update()
        
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