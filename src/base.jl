using LinearAlgebra
using Random
using Distributions

# kernel function
function rbf_kernel(x1, x2, l=1.0, sigma=1.0)
    return sigma^2 * exp(-0.5  * (sum((x1 .- x2).^2)) / l^2)    
end

function build_kernel_mtrx(X, kernel_fun, args...)
    n = length(X)
    K = zeros(n, n)
    for i in 1:n
        for j in 1:n
            K[i, j] = kernel_fun(X[i], X[j], args...)
        end
    end
    return K
end

function generate_synthetic_data(n=20, noise=0.1)
    Random.seed!(123)
    X = sort(2 * π * rand(n))
    y = sin.(X) + noise * randn(n)
    return X, y
end

function split_data(X, y, train_ratio=0.8)
    n = length(X)
    n_train = Int(floor(train_ratio * n))
    
    indices = randperm(n)
    train_indices = indices[1:n_train]
    test_indices = indices[n_train+1:end]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    return X_train, y_train, X_test, y_test
end