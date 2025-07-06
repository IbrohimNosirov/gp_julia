using LinearAlgebra
using Random
using Distributions

# check log EI gradient
# 0. gradient of k(x,x') = -sigma^2 exp(-0.5l^{-2} ||x - x'||^2)
# 1. gradient of mu(x)
# 2. gradient of sigma(x)

# gradient of k(x,x')
N = 10
X = randn(N, 2)
function rbf_kernel(x::Vector{Float64})
    x0 = [0.0, 0.0]
    lengthscale = 0.5
    variance = 1.0
    return variance*exp(-0.5*sum((x0 .- x).^2)/(lengthscale^2))
end

function grad_k(k::Function, x::Vector{Float64})
    lengthscale = 0.5
    grad_k = k(x)*(-1/(lengthscale^2)*x)
end

function finite_difference(k::Function, x::Vector{Float64})
    h = 1e-5
    x_plus = copy(x)
    x_minus = copy(x)
    x_plus .+= h/2 # * d where d = [1 1]
    x_minus .-= h/2

    return (k(x_plus) - k(x_minus))/h
end

print("gradient of k \n")
for i=1:N
    x = X[i, :]
    f_grad_k = grad_k(rbf_kernel, x)
    f_fd = finite_difference(rbf_kernel, x)
    # compute [1 1]^Tgrad_k(x) - finite difference approximation.
    print((f_grad_k[1] + f_grad_k[2] - f_fd), "\n")
end

print("gradient of mu is gradient of k times a constant inv(K_{xx}) y\n")
print("gradient of cov is gradient of 2*(grad_k^T K_{XX}^{-1}k_{xx*}).\n")
