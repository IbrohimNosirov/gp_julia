using LinearAlgebra
using Distributions
using Random
using Plots
using Optim
using StatsFuns
using DispatchDoctor: @stable
using BenchmarkTools
using Test

diff_fd(f, x=0.0; h=1e-6) = (f(x+h) - f(x-h))/(2h)

function kronecker_quasirand(d, N, start=0)
    φ = 1.0 + 1.0/d
    for k = 1:10
        gφ = φ^(d+1) - φ - 1
        dgφ = (d+1)*φ^d - 1
        φ -= gφ/dgφ
    end
    αs = [mod(1.0/φ^j, 1.0) for j=1:d]

    # Compute the quasi-random sequence
    Z = zeros(d, N)
    for j = 1:N
        for i = 1:d
            Z[i,j] = mod(0.5 + (start+j)*αs[i], 1.0)
        end
    end
    Z
end

# David's notes: https://www.cs.cornell.edu/courses/cs6241/2025sp/lec/2025-03-11.html
@stable function dist2(x :: AbstractVector{T}, y :: AbstractVector{T}) where {T}
    s = zero(T)
    for k = 1:length(x)
        dk = x[k] - y[k]
        s += dk * dk
    end
    s
end

@stable dist(x :: AbstractVector{T}, y :: AbstractVector{T}) where {T} = sqrt(dist2(x,y))

abstract type KernelContext end
# convenience function
(ctx :: KernelContext)(args ... ) = kernel(ctx, args ... )

abstract type RBFKernelContext{d} <: KernelContext end

ndims(::RBFKernelContext{d}) where {d} = d

@stable function Dφ_SE(s :: Float64)
    φ = exp(-s^2/2)
    dφ_div = -φ
    dφ = dφ_div*s
    Hφ = (-1 + s^2)*φ
    φ, dφ_div, dφ, Hφ
end

struct KernelSE{d} <: RBFKernelContext{d}
    l :: Float64
end

# tie the kernel context into operations I do all the time.
# squared exponential kernel.
φ_SE(s :: Float64) = exp(-s^2/2)
# pass kernel into Kernel context.
φ(:: KernelSE, s) = φ_SE(s)
Dφ(:: KernelSE, s) = Dφ_SE(s)
nhypers(:: KernelSE) = 1
getθ!(θ, ctx :: KernelSE) = θ[1]=ctx.l
updateθ!(ctx :: KernelSE{d}, θ) where {d} = ctx.l=θ[1]

kernel(ctx :: RBFKernelContext, x :: AbstractVector, y :: AbstractVector) =
φ(ctx, dist(x,y)/ctx.l)

# convenience function.
function getθ(ctx :: KernelContext)
    θ = zeros(nhypers(ctx))
    getθ!(θ, ctx)
    θ
end

function kernel_gθ!(g :: AbstractVector, ctx :: RBFKernelContext,
                    x :: AbstractVector, y :: AbstractVector, c=1.0)
    l = ctx.l
    s = dist(x,y)/l
    _, _, dφ, _ = Dφ(ctx, s)
    g[1] -= c * dφ * s / l
    g
end

function kernel_Hθ!(H :: AbstractMatrix, ctx :: RBFKernelContext,
                    x :: AbstractVector, y :: AbstractVector, c=1.0)
    l = ctx.l
    s = dist(x,y)/l
    _, _, dφ, Hφ = Dφ(ctx, s)
    H[1,1] += c*(Hφ*s + 2*dφ)*s/l^2
    H
end

function kernel_gx!(g :: AbstractVector, ctx :: RBFKernelContext,
                    x :: AbstractVector, y :: AbstractVector, c=1.0)
    l = ctx.l
    d = ndims(ctx)
    ρ = dist(x,y)
    s = ρ/l
    _, _, dφ, _ = Dφ(ctx, s)
    if ρ != 0.0
        dφ /= ctx.l
        C = c*dφ/ρ
        for i = 1:d
            g[i] += C*(x[i] - y[i])
        end
    end
    g
end

function kernel_Hx!(H :: AbstractMatrix, ctx :: RBFKernelContext,
                    x :: AbstractVector, y :: AbstractVector, c=1.0)
    l = ctx.l
    d = ndims(ctx)
    ρ = dist(x,y)
    s = ρ/l
    _, dφ_div, _, Hφ = Dφ(ctx, s)
    Hφ /= l^2
    dφ_div /= l^2
    for j = 1:d
        H[j,j] += c*dφ_div
    end
    if ρ != 0.0
        C = c*(Hφ - dφ_div)/ρ^2
        for j = 1:d
            xj, yj = x[j], y[j]
            for i = 1:d
                xi, yi = x[i], y[i]
                H[i,j] += C*(xj - yj)*(xi - yi)
            end
        end
    end
    H
end

# convenience functions
kernel_gθ_alloc(ctx :: RBFKernelContext,
                x :: AbstractVector, y :: AbstractVector) =
    kernel_gθ!(zeros(nhypers(ctx)), ctx, x, y)

kernel_Hθ_alloc(ctx :: RBFKernelContext,
                x :: AbstractVector, y :: AbstractVector) = 
    kernel_Hθ!(zeros(nhypers(ctx), nhypers(ctx)), ctx, x, y)

kernel_gx_alloc(ctx :: RBFKernelContext{d},
                x :: AbstractVector, y :: AbstractVector) where {d} =
    kernel_gx!(zeros(d), ctx, x, y)

kernel_Hx_alloc(ctx :: RBFKernelContext{d},
                x :: AbstractVector, y :: AbstractVector) where {d} =
    kernel_Hx!(zeros(d,d), ctx, x, y)

let
    function fd_check_Dφ(Dφ, s; kwargs ... )
        φ, dφ_div, dφ, Hφ = Dφ(s; kwargs ... )
        @test dφ_div*s ≈ dφ
        @test dφ ≈ diff_fd(s->Dφ(s; kwargs ... )[1], s) rtol=1e-6
        @test Hφ ≈ diff_fd(s->Dφ(s; kwargs ... )[3], s) rtol=1e-6
    end
    
    @testset "SE kernel derivative check" begin
        s = .123
        @testset "SE" fd_check_Dφ(Dφ_SE, s)
    end

    @testset "Kernel hyper derivatives" begin
        x = [0.1; 0.2]
        y = [0.8; 0.8]
        l = 0.123
        k_ctx(l) = KernelSE{2}(l)
        # convenience function to be able to call struct as a function
        k(l) = kernel(k_ctx(l), x, y)
        g(l) = kernel_gθ_alloc(k_ctx(l), x, y)[1]
        H(l) = kernel_Hθ_alloc(k_ctx(l), x, y)[1,1]
        @test g(l) ≈ diff_fd(k, l) rtol=1e-6
        @test H(l) ≈ diff_fd(g, l) rtol=1e-6
    end

    @testset "Kernel spatial derivatives" begin
        x = [0.2; 0.4]
        y = [0.3; 0.7]
        dx =[0.3; 0.5] # "what is the rate of change of x in the direction of this vector?"
        l = 0.32
        k_ctx = KernelSE{2}(l)
        # convenience function to be able to call struct as a function
        k(x) = kernel(k_ctx, x, y)
        g(x) = kernel_gx_alloc(k_ctx, x, y)
        H(x) = kernel_Hx_alloc(k_ctx, x, y)
        @test g(x)' * dx ≈ diff_fd(s->k(x + s*dx)) rtol=1e-6 
        @test H(x) * dx ≈ diff_fd(s->g(x + s*dx)) rtol=1e-6
    end
end

function kernel!(KXX :: AbstractMatrix, k :: KernelContext, X :: AbstractMatrix)
    for j = 1:size(X,2)
        xj = @view X[:,j]
        KXX[j,j] = k(xj, xj)
        for i = 1:j-1
            xi = @view X[:,i]
            kij = k(xi, xj)
            KXX[i,j] = kij
            KXX[j,i] = kij
        end
    end
    KXX
end

function kernel!(KXz :: AbstractVector, k :: KernelContext,
                 X :: AbstractMatrix, z :: AbstractVector)
    for i = 1:size(X,2)
        xi = @view X[:,i]
        KXz[i] = k(xi, z)
    end
    KXz
end

function kernel!(KXY :: AbstractMatrix, k :: KernelContext,
                 X :: AbstractMatrix, Y :: AbstractMatrix)
    for j = 1:size(Y,2)
        yj = @view Y[:,j]
        for i = 1:size(X,2)
            xi = @view X[:,i]
            KXY[i,j] = k(xi,yj)
        end
    end
    KXY
end

# convenience functions.
kernel_alloc(k :: KernelContext, X :: AbstractMatrix) =
    kernel!(zeros(size(X,2), size(X,2)), k, X)

kernel_alloc(k :: KernelContext, X :: AbstractMatrix, z :: AbstractVector) =
    kernel!(zeros(size(X,2)), k, X, z)

kernel_alloc(k :: KernelContext, X :: AbstractMatrix, Y :: AbstractMatrix) = 
    kernel!(zeros(size(X,2), size(Y,2)), k, X, Y)

let 
    Zk = kronecker_quasirand(2, 10)
    k_ctx = KernelSE{2}(1.0)
    Ktemp = zeros(10,10)
    Kvtemp = zeros(10)
    Zk1 = Zk[:,1]

    KXX1 = @time kernel_alloc(k_ctx, Zk)
    KXX2 = @time kernel_alloc(k_ctx, Zk, Zk)
    KXz2 = @time kernel_alloc(k_ctx, Zk, Zk[:,1])
    KXX2 = @time kernel!(Ktemp, k_ctx, Zk)
    KXX3 = @time kernel!(Ktemp, k_ctx, Zk, Zk)
    KXz2 = @time kernel!(Kvtemp, k_ctx, Zk, Zk1)
end

#=
Extended Cholesky:
Now that we have some data structures, we need a mechanism for making predictions.
    We do this by computing, storing, and adding to a Cholesky factorization.
        ∙ by default in Julia, Cholesky is stored in the upper triangle.
        ∙ BLAS symmetric rank-k update 'syrk' is an in-place call better optimized than
            A22 .-= R12'*R12.
        _____________
       |         |   |
       |    R    |R12|
       |_________|___|
       |    0    |R22|
       |_________|___|
=#

function extend_cholesky!(storage_mtrx::AbstractMatrix, n, m)
    #=
        storage_mtrx    a matrix with pre-Cholesky information, ready for in-place Cholesky.
        n::Integer      start of extension
        m::Integer      end of extension
    =#
    # Cholesky with space for extension
    R           = @view storage_mtrx[1:m, 1:m]
    # current Cholesky
    R11         = @view storage_mtrx[1:n, 1:n]
    # new rows with pre-Cholesky values
    A12 = @view storage_mtrx[1:n, n+1:m]
    A22 = @view storage_mtrx[n+1:m, n+1:m]

    
    ldiv!(UpperTriangular(R11)', A12)         # R12 = R11' \ A12
    BLAS.syrk!('U', 'T', -1.0, A12, 1.0, A22) # S = A22 - R12'*R12
    cholesky!(Symmetric(A22))

    # Return extended cholesky view
    Cholesky(UpperTriangular(R))
end

@testset "Check Cholesky extension" begin
    # Pre-generated random matrix
    A = [1.362     0.767029  0.991061  1.07994  1.35389;
         0.767029  1.55874   1.36436   1.5897   1.34834;
         0.991061  1.36436   1.49529   1.46581  1.43195;
         1.07994   1.5897    1.46581   1.97641  1.69469;
         1.35389   1.34834   1.43195   1.69469  2.19448]
    A_full = cholesky(A)

    Chol_1 = extend_cholesky!(A, 0, 3)
    @test Chol_1.U ≈ A_full.U[1:3,1:3]
    Chol_2 = extend_cholesky!(A, 3, 5)
    @test Chol_2.U ≈ A_full.U
end

@testset "Check tridiagonalization" begin

end

#=
Julia docs
   Constructors: It is good practice to provide as few inner constructor methods as possible:
   only those taking all arguments explicitly and enforcing essential error checking and
   transformation. Additional convenience constructor methods, supplying default values or
   auxiliary transformations, should be provided as outer constructors that call the inner
   constructors to do the heavy lifting. This separation is typically quite natural.
=#

struct GP
    X::Matrix{Float64}
    y::Vector{Float64}
    kernel::Function
    L::Matrix{Float64}
    lengthscale::Float64
    variance::Float64
    # check the sizes on X and y
    # check that size of L matches X
    function GP(X::Matrix{Float64}, y::Vector{Float64}, L::Matrix{Float64},
                kernel::Function, lengthscale::Float64, variance::Float64)
        return new(X, y, kernel, L, lengthscale, variance)
    end

    function GP(X::Matrix{Float64}, y::Vector{Float64}, kernel::Function,
            lengthscale::Float64, variance::Float64)
        K = kernel_matrix_compute(X, X, kernel, lengthscale, variance)
        L = cholesky(K).L
        return new(X, y, kernel, L, lengthscale, variance)
    end
end

#function predict(gp::GP, x_new::VecOrMat{Float64})
#    K_inv = gp.L' \ (gp.L \ gp.y)
#    K21 = kernel_matrix_compute(x_new, gp.X, gp.kernel, gp.lengthscale, gp.variance)
#    K22 = kernel_matrix_compute(x_new, x_new, gp.kernel, gp.lengthscale, gp.variance)
#    mu = K21 * K_inv
#    v = gp.L \ K21'
#    sigma2 = K22 - v' * v
#
#    return mu, diag(sigma2)
#end
#
#function kernel_matrix_compute(X1::VecOrMat{Float64}, X2::VecOrMat{Float64},
#k::Function, lengthscale::Float64, variance::Float64)
#    n1 = size(X1, 2)
#    n2 = size(X2, 2)
#    K = zeros(n1, n2)
#    for i in 1:n1
#        for j in 1:n2
#            K[i, j] = k(X1[:, i], X2[:, j], lengthscale, variance)
#        end
#    end
#
#    return K
#end
#
#function logEI(z::Float64)
#    function DψNLG0(z)
#        φz = normpdf(z)
#        Qz = normccdf(z)
#        Gz = φz - z*Qz
#        ψz = -log(Gz)
#        dψz = Qz/Gz
#        Hψz = (-φz*Gz + Qz^2)/Gz^2
#        ψz, dψz, Hψz
#    end
#
#    function DψNLG2(z)
#        # Approximate W by 20th convergent
#        W = 0.0
#        for k = 20:-1:1
#            W = k/(z + W)
#        end
#        ψz = log1p(z/W) + 0.5*(z^2 + log(2π))
#        dψz = 1/W
#        Hψz = (1 - W*(z + W))/W^2
#        ψz, dψz, Hψz
#    end
#    DψNLG(z) = if z < 6.0 DψNLG0(z) else DψNLG2(z) end
#end
#
#function get_Copt(gp :: GP)
#end
#
#function mean(gp :: GP, x :: AbstractVector)
#
#end
#
#function gx_mean(gp :: GP, x :: AbstractVector)
#end
#
#function Hx_mean(gp :: GP, x :: AbstractVector)
#end
#
#function var(gp :: GP, x :: AbstractVector)
#end
#
#function gx_var(gp :: GP, x :: AbstractVector)
#end
#
#function Hx_var(gp :: GP, x :: AbstractVector)
#end
#
#function Hgx_αNLEI(gp :: GP, x :: AbstractVector, y_best :: Float64)
#    Copt = getCopt(gp)
#    μ, gμ, Hμ = mean(gp, x), gx_mean(gp, x), Hx_mean(gp, x)
#    v, gv, Hv = Copt*var(gp, x), Copt*gx_var(gp, x), Cop*Hx_var(gp, x)
#
#    σ = sqrt(v)
#    gμs, Hμs = gμ/σ, Hμ/σ
#    gvs, Hvs = gv/(2v), Hv/v
#    
#    u = (μ - y_best)/σ
#    ψ, dψ, Hψ = logEI(u)
#
#    α = -log(σ) + ψ
#    dα = dψ*gμs - (1 + u*dψ)*gvs
#    Hα = -0.5*(1.0 + u*dψ)*Hvs + dψ*Hμs + Hψ*gμs*gμs' + (2.0 + u^2*Hψ + 3.0*u*dψ)*gvs*gvs'
#    -(u*Hψ + dψ)*(gμs*gvs' + gvs*gμs')
#
#    α, dα, Hα
#end
#
#@stable function optimize_EI(gp::GP, x_current :: AbstractVector, lo :: AbstractVector,
#        hi :: AbstractVector) 
#    y_best = minimum(gety(gp))
#    fun(x) = Hgx_αNLEI(gp, x, y_best)[1]
#    fun_g!(g, x) = copyto!(g, Hgx_αNLEI(gp, x, y_best)[2])
#    fun_H!(g, x) = copyto!(g, Hgx_αNLEI(gp, x, y_best)[3])
#    df = TwiceDifferentiable(fun, fun_g!, fun_H!, x0)
#    dfc = TwiceDifferentiableConstraints(lo, hi)
#    res = optimize(df, dfc, x0, IPNewton())
#end
#
#@stable function acquire_next_point(gp::GP, lo :: AbstractVector, hi :: AbstractVector;
#        nstarts = 10, verbose=true)
#    y_next = Inf
#    x_next = [0.0; 0.0]
#    for j = 1:10
#        z = lo + (hi-lo).*rand(length(lo))
#        res = optimize_EI(gp, z, [0.0; 0.0], [1.0; 1.0])
#        if verbose
#            println("From $z: $(Optim.minimum(res)) at $(Optim.minimizer(res))")
#        end
#        if Optim.minimum(res) < bestα
#            y_next = Optim.minimum(res)
#            x_next[:] = Optim.minimizer(res)
#        end
#    return x_next, y_next
#end
#
#function surrogate_model_update!(gp::GP, x::Vector{Float64}, y::Float64)
#    X_new = [gp.X x]
#    y_new = [gp.y; y]
#    n_points = size(X_new, 2)
#    
#    if n_points % 5 == 0
#        lengthscale_new, variance_new = optimize_hyperparameters(X_new, y_new,
#        gp.kernel, gp.lengthscale, gp.variance)
#
#        K = kernel_matrix_compute(X_new, X_new, gp.kernel, lengthscale_new, variance_new)
#        L_new = cholesky(K + 1e-10*I(size(K,1))).L
#        L_new = Matrix(L_new)
#    else
#        K12 = kernel_matrix_compute(gp.X, reshape(x, :, 1), gp.kernel, gp.lengthscale,
#        gp.variance)
#        K22 = kernel_matrix_compute(reshape(x, :, 1), reshape(x, :, 1), gp.kernel,
#        gp.lengthscale, gp.variance)
#        
#        L12 = gp.L \ K12
#        L22 = sqrt(K22 - L12' * L12) .+ 1e-6
#        L_new = [gp.L zeros(Float64, size(gp.L, 1), 1); 
#                    reshape(L12', 1, :) L22]
#        
#        lengthscale_new = gp.lengthscale
#        variance_new = gp.variance
#    end
#    return GP(X_new, y_new, L_new, gp.kernel, lengthscale_new, variance_new)
#end
#
#function optimize_hyperparameters(X, y, kernel, lengthscale, variance; method=LBFGS())
#    initial_params = [lengthscale, variance]
#    lower_bounds = [1e-6, 1e-6]
#    upper_bounds = [10.0, 10.0]
#
#    function objective(params)
#        return -log_marginal_likelihood(X, y, kernel, params)
#    end
#
#    result = Optim.optimize(objective, lower_bounds, upper_bounds, initial_params,
#    Fminbox(method))
#    opt_params = Optim.minimizer(result)
#    opt_lengthscale = opt_params[1]
#    opt_variance = opt_params[2]
#    
#    return opt_lengthscale, opt_variance
#end
#
#function log_marginal_likelihood(X::Matrix{Float64}, y::Vector{Float64}, kernel::Function,
#        params::Vector{Float64})
#    lengthscale = params[1]
#    variance = params[2]
#    K = kernel_matrix_compute(X, X, kernel, lengthscale, variance)
#    n = size(K, 1)
#    L = cholesky(K + 1e-6*I(size(K,1))).L
#    α = L' \ (L \ y)
#    
#    return -0.5 * dot(y, α) - sum(log.(diag(L))) - 0.5 * n * log(2π)
#end
#
#function BO_loop(f::Function, bounds::Matrix{Float64}, n_iterations::Int; n_init::Int=5)
#    dimensions = size(bounds, 1)
#    lengthscale = 0.5
#    variance = 1.2
#
#    X_init = zeros(dimensions, n_init)
#    for d in 1:dimensions
#        X_init[d, :] = bounds[1, d] .+ (bounds[2, d] - bounds[1, d]) * rand(n_init)
#    end
#    y_init = [f(X_init[:, i]) for i in 1:n_init]
#
#    gp = GP(X_init, y_init, rbf_kernel, lengthscale, variance)
#
#    X_all = copy(X_init)
#    y_all = copy(y_init)
#    y_best = minimum(y_init)
#    x_best = X_init[:, argmin(y_init)]
#
#    history = [(x_best, y_best)]
#    x_current = x_best
#
#    for i in 1:n_iterations
#        x_current, ei = acquire_next_point(gp, x_current, y_best)
#        y_current = f(x_current)
#        gp = surrogate_model_update!(gp, x_current, y_current)
#        X_all = [X_all x_current]
#        y_all = [y_all; y_current]
#        if y_current < y_best
#            y_best = y_current
#            x_best = x_current
#            push!(history, (x_best, y_best))
#        end
#
#        println("Iteration $i: x = $x_current, f(x) = $y_current, Best = $y_best")
#    end
#    lengthscale_final = gp.lengthscale
#    variance_final = gp.variance
#    println("Final lengthscale $lengthscale_final, final variance $variance_final")
#
#    return x_best, y_best, history, gp
#end
