using LinearAlgebra
using Distributions
using Random
using Plots
using Optim

# David's notes: https://www.cs.cornell.edu/courses/cs6241/2025sp/lec/2025-03-11.html
function rbf_kernel(x1::Vector{Float64}, x2::Vector{Float64}, lengthscale::Float64,
        variance::Float64)
    return variance*exp(-0.5*sum((x1 .- x2).^2)/(lengthscale^2))
end

struct GP
    X::Matrix{Float64}
    y::Vector{Float64}
    kernel::Function
    L::Matrix{Float64}
    lengthscale::Float64
    variance::Float64

    function GP(X::Matrix{Float64}, y::Vector{Float64}, kernel::Function,
            lengthscale::Float64, variance::Float64)
        K = kernel_matrix_compute(X, X, kernel, lengthscale, variance)
        L = cholesky(K).L
        return new(X, y, kernel, L, lengthscale, variance)
    end

    function GP(X::Matrix{Float64}, y::Vector{Float64}, L::Matrix{Float64},
                kernel::Function, lengthscale::Float64, variance::Float64)
        return new(X, y, kernel, L, lengthscale, variance)
    end
end

function predict(gp::GP, x_new::VecOrMat{Float64})
    K_inv = gp.L' \ (gp.L \ gp.y)
    K21 = kernel_matrix_compute(x_new, gp.X, gp.kernel, gp.lengthscale, gp.variance)
    K22 = kernel_matrix_compute(x_new, x_new, gp.kernel, gp.lengthscale, gp.variance)
    mu = K21 * K_inv
    v = gp.L \ K21'
    sigma2 = K22 - v' * v

    return mu, diag(sigma2)
end

function kernel_matrix_compute(X1::VecOrMat{Float64}, X2::VecOrMat{Float64},
k::Function, lengthscale::Float64, variance::Float64)
    n1 = size(X1, 2)
    n2 = size(X2, 2)
    K = zeros(n1, n2)
    for i in 1:n1
        for j in 1:n2
            K[i, j] = k(X1[:, i], X2[:, j], lengthscale, variance)
        end
    end

    return K
end

function acquire_next_point(gp::GP, x_current::AbstractVector, y_best::Float64)
    function EI(x::AbstractVector, y_best::Float64, predict::Function)
        mean, var = predict(gp, x)
        std = 0
        if var[1] > 1e-14
            std = sqrt(var[1])
            z = (y_best - mean[1]) / std
        else
            z = 0
        end
        cdf_z = cdf(Normal(), z)
        pdf_z = pdf(Normal(), z)

        (y_best - mean[1])*cdf_z + std*pdf_z
        return 1
    end

    # wrt x1; fix x2 
    function k_g!(storage::AbstractVector, x1::AbstractVector, x2::AbstractVector)
        storage = k(x1, x2, gp.lengthscale, gp.variance) * -1/(gp.lengthscale^2) * x1
    end

    function logEI_g!(storage::AbstractVector, x::Vector{Float64}, EI::Function,
            k_g!::Function)
        # output should be a vector
        EI_value = EI(x, y_best, predict)
        # TODO: Populate x_k_g with all kernel evaluations
        N = size(gp.X)[2]
        grad_x_k = zeros(size(x)[1],N)
        for i=1:N
            k_g!(grad_x_k[:,i], x, X[:, i])
        end
        pdf_z = pdf(Normal(), z)
        cdf_z = cdf(Normal(), z)
        grad_mu = grad_x_k * gp.L' \ (gp.L \ gp.y)
        grad_sigma = -2 * grad_x_k' * (gp.L \ k)

        storage = (-cdf_z * grad_mu + pdf_z * grad_sigma)/EI_value
    end

    storage = zeros(size(x_current))
    logEI_closure(x) = log(EI(storage, EI, k_g!, predict, x))
    EI_closure(x) = EI(x, y_best, predict)
    logEI_g_closure!(x) = logEI_g!(storage, x, EI, k_g!, predict)
    y_next = Optim.optimize(logEI_closure, logEI_g_closure!, x_current, LBFGS())
    x_next = Optim.minimizer(y_next)

    return x_next, y_next
end

function surrogate_model_update!(gp::GP, x::Vector{Float64}, y::Float64)
    X_new = [gp.X x]
    y_new = [gp.y; y]
    n_points = size(X_new, 2)
    
    if n_points % 5 == 0
        lengthscale_new, variance_new = optimize_hyperparameters(X_new, y_new,
        gp.kernel, gp.lengthscale, gp.variance)

        K = kernel_matrix_compute(X_new, X_new, gp.kernel, lengthscale_new, variance_new)
        L_new = cholesky(K + 1e-10*I(size(K,1))).L
        L_new = Matrix(L_new)
    else
        K12 = kernel_matrix_compute(gp.X, reshape(x, :, 1), gp.kernel, gp.lengthscale,
        gp.variance)
        K22 = kernel_matrix_compute(reshape(x, :, 1), reshape(x, :, 1), gp.kernel,
        gp.lengthscale, gp.variance)
        
        L12 = gp.L \ K12
        L22 = sqrt(K22 - L12' * L12) .+ 1e-6
        L_new = [gp.L zeros(Float64, size(gp.L, 1), 1); 
                    reshape(L12', 1, :) L22]
        
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
    opt_params = Optim.minimizer(result)
    opt_lengthscale = opt_params[1]
    opt_variance = opt_params[2]
    
    return opt_lengthscale, opt_variance
end

function log_marginal_likelihood(X::Matrix{Float64}, y::Vector{Float64}, kernel::Function,
        params::Vector{Float64})
    lengthscale = params[1]
    variance = params[2]
    K = kernel_matrix_compute(X, X, kernel, lengthscale, variance)
    n = size(K, 1)
    L = cholesky(K + 1e-6*I(size(K,1))).L
    α = L' \ (L \ y)
    
    return -0.5 * dot(y, α) - sum(log.(diag(L))) - 0.5 * n * log(2π)
end

function BO_loop(f::Function, bounds::Matrix{Float64}, n_iterations::Int; n_init::Int=5)
    dimensions = size(bounds, 1)
    lengthscale = 0.5
    variance = 1.2

    X_init = zeros(dimensions, n_init)
    for d in 1:dimensions
        X_init[d, :] = bounds[1, d] .+ (bounds[2, d] - bounds[1, d]) * rand(n_init)
    end
    y_init = [f(X_init[:, i]) for i in 1:n_init]

    gp = GP(X_init, y_init, rbf_kernel, lengthscale, variance)

    X_all = copy(X_init)
    y_all = copy(y_init)
    y_best = minimum(y_init)
    x_best = X_init[:, argmin(y_init)]

    history = [(x_best, y_best)]
    x_current = x_best

    for i in 1:n_iterations
        x_current, ei = acquire_next_point(gp, x_current, y_best)
        y_current = f(x_current)
        gp = surrogate_model_update!(gp, x_current, y_current)
        X_all = [X_all x_current]
        y_all = [y_all; y_current]
        if y_current < y_best
            y_best = y_current
            x_best = x_current
            push!(history, (x_best, y_best))
        end

        println("Iteration $i: x = $x_current, f(x) = $y_current, Best = $y_best")
    end
    lengthscale_final = gp.lengthscale
    variance_final = gp.variance
    println("Final lengthscale $lengthscale_final, final variance $variance_final")

    return x_best, y_best, history, gp
end
