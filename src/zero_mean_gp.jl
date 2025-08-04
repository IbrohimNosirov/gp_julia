using LinearAlgebra
using Distributions
using Random
using Plots
using Optim
using StatsFuns
using DispatchDoctor: @stable
using Test

#David: why does only x=0.0 work?
diff_fd(f, x=0.0; h=1e-6) = (f(x+h) - f(x-h))/(2h) 

sample_eval(f, X :: AbstractMatrix) = [f(x) for x in eachcol(X)]
sample_eval2(f, X :: AbstractMatrix) = [f(x...) for x in eachcol(X)]

function test_setup2d(f, n)
    Zk = kronecker_quasirand(2,n)
    y = sample_eval2(f, Zk)
    Zk, y
end

test_setup2d(f) = test_setup2d(f,10)

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

# David's notes:
# https://www.cs.cornell.edu/courses/cs6241/2025sp/lec/2025-03-11.html
@stable function dist2(x :: AbstractVector{T}, y :: AbstractVector{T}) where {T}
    s = zero(T)
    for k = 1:length(x)
        dk = x[k] - y[k]
        s += dk * dk
    end
    s
end

dist(x :: AbstractVector{T}, y :: AbstractVector{T}) where {T} =
    sqrt(dist2(x,y))

abstract type KernelContext end
# convenience function
(ctx :: KernelContext)(args ... ) = kernel_func(ctx, args ... )

abstract type RBFKernelContext{d} <: KernelContext end

ndims(::RBFKernelContext{d}) where {d} = d

function Dφ_SE(s :: Float64)
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

kernel_func(ctx :: RBFKernelContext, x :: AbstractVector, y :: AbstractVector) =
    φ(ctx, dist(x,y)/ctx.l)

# convenience function.
function getθ(ctx :: KernelContext)
    θ = zeros(nhypers(ctx))
    getθ!(θ, ctx)
    θ
end

function kernel!(KXX :: AbstractMatrix, k_ctx :: KernelContext,
                 X :: AbstractMatrix, η :: Real = 0.0)
    for j = 1:size(X,2)
        xj = @view X[:,j]
        KXX[j,j] = k_ctx(xj, xj) + η
        for i = 1:j-1
            xi = @view X[:,i]
            kij = k_ctx(xi, xj)
            KXX[i,j] = kij
            KXX[j,i] = kij
        end
    end
    KXX
end

function kernel!(KXz :: AbstractVector, k_ctx :: KernelContext,
                 X :: AbstractMatrix, z :: AbstractVector)
    for i = 1:size(X,2)
        xi = @view X[:,i]
        KXz[i] = k_ctx(xi, z)
    end
    KXz
end

function kernel!(KXY :: AbstractMatrix, k_ctx :: KernelContext,
                 X :: AbstractMatrix, Y :: AbstractMatrix)
    for j = 1:size(Y,2)
        yj = @view Y[:,j]
        for i = 1:size(X,2)
            xi = @view X[:,i]
            KXY[i,j] = k_ctx(xi,yj)
        end
    end
    KXY
end

# convenience functions.
kernel_alloc(k_ctx :: KernelContext, X :: AbstractMatrix, η :: Real = 0.0) =
    kernel!(zeros(size(X,2), size(X,2)), k_ctx, X, η)

kernel_alloc(k_ctx :: KernelContext, X :: AbstractMatrix, z :: AbstractVector) =
    kernel!(zeros(size(X,2)), k_ctx, X, z)

kernel_alloc(k_ctx :: KernelContext, X :: AbstractMatrix, Y :: AbstractMatrix) = 
    kernel!(zeros(size(X,2), size(Y,2)), k_ctx, X, Y)

@testset "Compare kernel matrix constructors" begin
    Zk = kronecker_quasirand(2,10)
    k_ctx = KernelSE{2}(1.0)

    # Comprehension-based eval
    KXX1 = [k_ctx(x,y) for x in eachcol(Zk), y in eachcol(Zk)]
    KXz1 = [k_ctx(x,Zk[:,1]) for x in eachcol(Zk)]

    # Dispatch through call mechanism
    KXX2 = kernel_alloc(k_ctx, Zk)
    KXX3 = kernel_alloc(k_ctx, Zk, Zk)
    KXz2 = kernel_alloc(k_ctx, Zk, Zk[:,1])

    @test KXX1 ≈ KXX2
    @test KXX1 ≈ KXX3
    @test KXz1 ≈ KXX1[:,1]
    @test KXz2 ≈ KXX1[:,1]
end

let 
    Zk = kronecker_quasirand(2, 10)
    k_ctx = KernelSE{2}(1.0)
    Ktemp = zeros(10,10)
    Kvtemp = zeros(10)
    Zk1 = Zk[:,1]

    KXX1 = @time kernel_alloc(k_ctx, Zk)
    KXX2 = @time kernel_alloc(k_ctx, Zk, Zk)
    KXz2 = @time kernel_alloc(k_ctx, Zk, Zk[:,1])
    @time kernel!(Ktemp, k_ctx, Zk)
    @time kernel!(Ktemp, k_ctx, Zk, Zk)
    @time kernel!(Kvtemp, k_ctx, Zk, Zk1)
end

function kernel_gθ!(g :: AbstractVector, ctx :: RBFKernelContext,
                    x :: AbstractVector, y :: AbstractVector, c=1.0)
    l = ctx.l
    s = dist(x,y)/l
    _, _, dφ, _ = Dφ(ctx, s)
    g[1] -= c * dφ * s / l # David: why g[1]? because there is only 1?
    g
end

function kernel_Hθ!(H :: AbstractMatrix, ctx :: RBFKernelContext,
                    x :: AbstractVector, y :: AbstractVector, c=1.0)
    l = ctx.l
    s = dist(x,y)/l
    _, _, dφ, Hφ = Dφ(ctx, s)
    H[1,1] += c*(Hφ*s + 2*dφ)*s/l^2 # David: why H[1,1]? because there is 1?
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
        k(l) = kernel_func(k_ctx(l), x, y)
        g(l) = kernel_gθ_alloc(k_ctx(l), x, y)[1]
        H(l) = kernel_Hθ_alloc(k_ctx(l), x, y)[1,1]
        @test g(l) ≈ diff_fd(k, l) rtol=1e-6
        @test H(l) ≈ diff_fd(g, l) rtol=1e-6
    end

    @testset "Kernel spatial derivatives" begin
        x, y, dx  = [0.2; 0.4], [0.3; 0.7], [0.3; 0.5]
        l = 0.32
        k_ctx = KernelSE{2}(l)
        # convenience function to be able to call struct as a function
        k(x) = kernel_func(k_ctx, x, y)
        g(x) = kernel_gx_alloc(k_ctx, x, y)
        H(x) = kernel_Hx_alloc(k_ctx, x, y)
        @test g(x)' * dx ≈ diff_fd(s->k(x + s*dx)) rtol=1e-6 
        @test H(x) * dx ≈ diff_fd(s->g(x + s*dx)) rtol=1e-6
    end
end

#=
Extended Cholesky:
Now that we have some data structures, we need a mechanism for making
    predictions, running the optimization, and so forth.
    Julia comes with a Cholesky function, (cholesky!). We'll need to compute,
        store, and extend the kernel Cholesky data structure.
    There are two additional things to remember for extending
        ∙ by default in Julia, Cholesky is stored in the upper triangle.
        ∙ BLAS symmetric rank-k update 'syrk' is an in-place call better
        optimized than A22 .-= R12'*R12.
        _____________
       |         |   |
       |    R    |R12|
       |_________|___|
       |    0    |R22|
       |_________|___|
=#

kernel_cholesky(ctx :: KernelContext, X :: AbstractMatrix) =
    cholesky!(kernel_alloc(ctx, X))

kernel_cholesky!(KXX :: AbstractMatrix, ctx :: KernelContext,
                 X :: AbstractMatrix) =
    cholesky!(kernel!(KXX, ctx, X))

kernel_cholesky(ctx :: KernelContext, X :: AbstractMatrix, η :: Real) =
    cholesky!(kernel_alloc(ctx, X, η))

kernel_cholesky!(KXX :: AbstractMatrix, ctx :: KernelContext,
                 X :: AbstractMatrix, η :: Real) =
    cholesky!(kernel!(KXX, ctx, X, η))

let
    Zk = kronecker_quasirand(2, 10)
    k_ctx = KernelSE{2}(1.0)
    Ktemp = zeros(10,10)
    Kvtemp = zeros(10)
    Zk1 = Zk[:,1]
    η = 1e-8

    KXX = @time kernel_cholesky(k_ctx, Zk)
    @time kernel_cholesky!(Ktemp, k_ctx, Zk)
    KXX2 = @time kernel_cholesky(k_ctx, Zk, η)
    @time kernel_cholesky!(Ktemp, k_ctx, Zk, η)
end

function extend_cholesky!(storage_mtrx::AbstractMatrix, n, m)
    #=
        storage_mtrx    a matrix with pre-Cholesky information, ready for
                            in-place Cholesky.
        n::Integer      start of extension
        m::Integer      end of extension
    =#
    # Cholesky with space for extension
    R = @view storage_mtrx[1:m, 1:m]
    # current Cholesky
    R11 = @view storage_mtrx[1:n, 1:n]
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

#= David: Tridiagonalization. I don't have a solid reason to do this right now,
    so I'm going to
    move on for the time-being.
=#

function eval_GP(KC :: Cholesky, ctx :: KernelContext, X :: AbstractMatrix,
    c :: AbstractVector, z :: AbstractVector)
    kXz = kernel_alloc(ctx, X, z)
    μ = dot(kXz,c)
    rXz = ldiv!(KC.L, kXz)
    σ = sqrt(kernel_func(ctx, z, z) - rXz'*rXz)
    μ, σ
end

let
    # Set up sample points and test function
    testf(x,y) = x^2 + y
    Zk, y = test_setup2d(testf)
    ctx = KernelSE{2}(1.0)

    # Form kernel Cholesky and weights
    KC = kernel_cholesky(ctx, Zk)
    c = KC\y

    # Evaluate true function and GP at a test point
    z = [0.456; 0.436]
    fz = testf(z...)
    μz, σz = eval_GP(KC, ctx, Zk, c, z)

    # Compare GP to true function
    zscore = (fz-μz)/σz
    println("""
        True value:       $fz
        Posterior mean:   $μz
        Posterior stddev: $σz
        z-score:          $zscore
        """)
end

#=
Julia docs
   Constructors: It is good practice to provide as few inner constructor methods
   as possible: only those taking all arguments explicitly and enforcing
   essential error checking and transformation. Additional convenience
   constructor methods, supplying default values or auxiliary transformations,
   should be provided as outer constructors that call the inner constructors to
   do the heavy lifting. This separation is typically quite natural.
=#

# TODO: make dθ_kernel! I don't have a solid reason for this, but it's in the
    # notes, so I'm going to skip for the moment.

struct GPPContext{T <: KernelContext}
    ctx :: T
    η :: Float64
    Xstore  :: Matrix{Float64}
    Kstore  :: Matrix{Float64}
    cstore  :: Vector{Float64}
    ystore  :: Vector{Float64}
    scratch :: Matrix{Float64}
    n :: Integer
end

getX(gp :: GPPContext) = view(gp.Xstore,:,1:gp.n)
getc(gp :: GPPContext) = view(gp.cstore,1:gp.n)
gety(gp :: GPPContext) = view(gp.ystore,1:gp.n)
getK(gp :: GPPContext) = view(gp.Kstore,1:gp.n,1:gp.n)
getKC(gp :: GPPContext) = Cholesky(UpperTriangular(getK(gp)))
capacity(gp :: GPPContext) = length(gp.ystore)
getXrest(gp :: GPPContext) = view(gp.Xstore,:,gp.n+1:capacity(gp))
getyrest(gp :: GPPContext) = view(gp.ystore,gp.n+1:capacity(gp))
getXrest(gp :: GPPContext, m) = view(gp.Xstore,:,gp.n+1:gp.n+m)
getyrest(gp :: GPPContext, m) = view(gp.ystore,gp.n+1:gp.n+m)

function GPPContext(ctx :: KernelContext, η :: Float64, capacity)
    d = ndims(ctx)
    Xstore  = zeros(d, capacity)
    Kstore  = zeros(capacity, capacity)
    cstore  = zeros(capacity)
    ystore  = zeros(capacity)
    scratch = zeros(capacity,max(d+1,3))
    GPPContext(ctx, η, Xstore, Kstore, cstore, ystore, scratch, 0)
end

refactor!(gp :: GPPContext) = kernel_cholesky!(getK(gp), gp.ctx, getX(gp), gp.η)
resolve!(gp :: GPPContext) = ldiv!(getKC(gp), copyto!(getc(gp), gety(gp)))

function add_points!(gp :: GPPContext, m)
    n = gp.n + m
    if gp.n > capacity(gp)
        error("proposed points exceed GP context capacity.")
    end
    
    # Create new object (same storage)
    gpnew = GPPContext(gp.ctx, gp.η, gp.Xstore, gp.Kstore,
                       gp.cstore, gp.ystore, gp.scratch, n)

    #Refactor (if start from 0) or extend Cholesky (if partly done)
    if gp.n == 0
        refactor!(gpnew)
    else
        X1, X2 = getX(gp), getXrest(gp, m)
        R11 = getK(gp)
        K12 = view(gp.Kstore, 1:gp.n, gp.n+1:n)
        K22 = view(gp.Kstore, gp.n+1:n, gp.n+1:n)
        kernel!(K12, gp.ctx, X1, X2)
        kernel!(K22, gp.ctx, X2, gp.η)
        ldiv!(UpperTriangular(R11)', K12)
        BLAS.syrk!('U', 'T', -1.0, K12, 1.0, K22)
        cholesky!(Symmetric(K22))
    end
    
    # Update c
    resolve!(gpnew)

    gpnew
end

function add_points!(gp :: GPPContext, X :: AbstractMatrix, y :: AbstractVector)
    m = length(y)
    @assert size(X,2) == m "Inconsistent number of points and number of values."
    copy!(getXrest(gp, m), X)
    copy!(getyrest(gp, m), y)
    add_points!(gp, m)
end

function add_point!(gp :: GPPContext, x :: AbstractVector, y :: Float64)
    add_points!(gp, reshape(x, length(x), 1), [y])
end

function GPPContext(ctx :: KernelContext, η :: Float64,
                    X :: Matrix{Float64}, y :: Vector{Float64})
    d, n = size(X)
    @assert d == ndims(ctx) "Mismatch in dimensions of X and kernel."
    gp = GPPContext(ctx, η, n)
    copy!(gp.Xstore, X)
    copy!(gp.ystore, y)
    add_points!(gp, n)
end

function remove_points!(gp :: GPPContext, m)
    @assert m <= gp.n "Cannot remove $m > $(gp.n) points."
    gpnew = GPPContext(gp.ctx, gp.η, gp.Xstore, gp.Kstore,
                       gp.cstore, gp.ystore, gp.scratch, gp.n-m)
   resolve!(gpnew)
   gpnew
end

function mean(gp :: GPPContext, z :: AbstractVector)
    ctx, X, c = gp.ctx, getX(gp), getc(gp)
    d, n = size(X)
    sz = 0.0
    for j = 1:n
        xj = @view X[:,j]
        sz += c[j]*kernel_func(ctx, z, xj)
    end
    sz
end

function mean_gx!(gsz :: AbstractVector, gp :: GPPContext, z :: AbstractVector)
    ctx, X, c = gp.ctx, getX(gp), getc(gp)
    d, n = size(X)
    for j = 1:n
        xj = @view X[:,j]
        kernel_gx!(gsz, ctx, z, xj, c[j])
    end
    gsz
end

function mean_gx(gp :: GPPContext, z :: AbstractVector)
    d = ndims(gp.ctx)
    mean_gx!(zeros(d), gp, z)
end

function mean_Hx!(Hsz :: AbstractMatrix, gp :: GPPContext, z :: AbstractVector)
    ctx, X, c = gp.ctx, getX(gp), getc(gp)
    d, n = size(X)
    for j = 1:n
        xj = @view X[:,j]
        kernel_Hx!(Hsz, ctx, z, xj, c[j])
    end
    Hsz
end

function mean_Hx(gp :: GPPContext, z :: AbstractVector)
    d = ndims(gp.ctx)
    mean_Hx!(zeros(d,d), gp, z)
end

function var(gp :: GPPContext, z :: AbstractVector)
    kXz = view(gp.scratch,1:gp.n,1)
    kernel!(kXz, gp.ctx, getX(gp), z)
    L = getKC(gp).L
    v = ldiv!(L, kXz)
    kernel_func(gp.ctx,z,z) - v'*v
end

function var_gx!(g :: AbstractVector, gp :: GPPContext, z :: AbstractVector)
    X, KC, ctx = getX(gp), getKC(gp), gp.ctx
    d, n = size(X)
    kXz  = view(gp.scratch,1:n,1)
    gkXz = view(gp.scratch,1:n,2:d+1)
    gkXz[:] .= 0.0
    for j = 1:n
        xj = @view X[:,j]
        kXz[j] = kernel_func(ctx, z, xj)
        kernel_gx!(view(gkXz,j,:), ctx, z, xj)
    end
    w = ldiv!(KC,kXz)
    mul!(g, gkXz', w, -2.0, 0.0)
end

function var_gx(gp :: GPPContext, z :: AbstractVector)
    d = ndims(gp.ctx)
    var_gx!(zeros(d), gp, z)
end

function var_Hx!(H :: AbstractMatrix, gp :: GPPContext, z :: AbstractVector)
    X, KC, ctx = getX(gp), getKC(gp), gp.ctx
    d, n = size(X)
    kXz  = view(gp.scratch,1:n,1)
    gkXz = view(gp.scratch,1:n,2:d+1)
    gkXz[:] .= 0.0
    for j = 1:n
        xj = @view X[:,j]
        kXz[j] = kernel_func(ctx, z, xj)
        kernel_gx!(view(gkXz,j,:), ctx, z, xj)
    end
    w = ldiv!(KC,kXz)
    invL_gkXz = ldiv!(KC.L, gkXz)
    H[:] .= 0.0
    for j = 1:n
        xj = @view X[:,j]
        kernel_Hx!(H, ctx, z, xj, w[j])
    end
    mul!(H, invL_gkXz', invL_gkXz, -2.0, -2.0)
end

function var_Hx(gp :: GPPContext, z :: AbstractVector)
    d = ndims(gp.ctx)
    var_Hx!(zeros(d,d), gp, z)
end

let
    testf(x,y) = x^2+y
    Zk, y = test_setup2d(testf)
    ctx = KernelSE{2}(1.0)
    gp = GPPContext(ctx, 0.0, Zk, y)

    z = [0.456; 0.456]
    fz = testf(z...)
    μz, σz = mean(gp, z), sqrt(var(gp,z))
    zscore = (fz-μz)/σz
    println("""
        True value:       $fz
        Posterior mean:   $μz
        Posterior stddev: $σz
        z-score:          $zscore
        """)
end

@testset "Update points and kernel in GPPContext" begin
    testf(x,y) = x^2+y
    Zk, y = test_setup2d(testf)
    ctx = KernelSE{2}(1.0)
    ctx2 = KernelSE{2}(0.8)

    gp1 = GPPContext(ctx, 0.0, Zk, y)
    gpt = GPPContext(ctx, 0.0, 10)
    gpt = add_points!(gpt, Zk, y)
    @test getc(gpt) ≈ getc(gp1)

    gpt = remove_points!(gpt, 2)
    gp3 = GPPContext(ctx, 0.0, Zk[:,1:end-2], y[1:end-2])
    @test getc(gpt) ≈ getc(gp3)

    gpt = add_points!(gpt, Zk[:,end-1:end], y[end-1:end])
    @test getc(gpt) ≈ getc(gp1)
end

@testset "Predictive mean and variance derivatives" begin
    Zk, y = test_setup2d((x,y) -> x^2 + cos(3*y))
    z, dz = [0.47; 0.47], [0.132; 0.0253]
    gp = GPPContext(KernelSE{2}(0.5), 1e-8, Zk, y)
    @test mean_gx(gp,z)'*dz ≈ diff_fd(s->mean(gp,z+s*dz))    rtol=1e-6
    @test mean_Hx(gp,z)*dz  ≈ diff_fd(s->mean_gx(gp,z+s*dz)) rtol=1e-6
    @test var_gx(gp,z)'*dz  ≈ diff_fd(s->var(gp,z+s*dz))     rtol=1e-6
    @test var_Hx(gp,z)*dz   ≈ diff_fd(s->var_gx(gp,z+s*dz))  rtol=1e-6
end

@testset "Test gradients " begin
    # David: is a second variational just a different direction vector?
    A0, δA, ΔA, ΔδA = randn(10,10), rand(10,10), rand(10,10), rand(10,10)

    δinv(A,δA) = -A\δA/A # ((A') \ (-A\δA)')' rdiv!(-A\δA,A) doesn't work? David
    invAδA, invAΔA, invAΔδA = A0\δA, A0\ΔA, A0\ΔδA
    ΔδinvA = (invAδA*invAΔA + invAΔA*invAδA - invAΔδA)/A0

    @test δinv(A0,δA) ≈ diff_fd(s->inv(A0+s*δA)) rtol=1e-6
    @test ΔδinvA ≈ diff_fd(s->δinv(A0+s*ΔA, δA+s*ΔδA)) rtol=1e-6

    V = randn(10,10)
    A = V*Diagonal(1.0.+rand(10))/V
    δA, ΔA, ΔδA = randn(10,10), randn(10,10), randn(10,10)

    δlogdet(A, δA) = tr(A\δA)
    Δδlogdet(A, δA, ΔA, ΔδA) = tr(A\ΔδA)-tr((A\ΔA)*(A\δA))

    @test δlogdet(A,δA) ≈ dot(inv(A'), δA)
    @test δlogdet(A,δA) ≈ diff_fd(s->log(det(A+s*δA))) rtol=1e-6
    @test Δδlogdet(A,δA,ΔA,ΔδA) ≈ diff_fd(s->δlogdet(A+s*ΔA,δA+s*ΔδA)) rtol=1e-6
end

# why is δ[logdet(A)] = δ[det(A)]/det(A)?

function nll(KC :: Cholesky, c :: AbstractVector, y :: AbstractVector)
    n = length(c)
    φ = (dot(c,y) + n*log(2π))/2
    for k = 1:n
        φ += log(KC.U[k,k])
    end
    φ
end

# convenience function
nll(gp :: GPPContext) = nll(getKC(gp), getc(gp), gety(gp))

function nll_gθ!(g :: AbstractVector, gp :: GPPContext, invK :: AbstractMatrix)
    # fast for loop. Once you choose a column, use it up completely.
    # recall that there is only 1 hyperparameter, lengthscale.
    ctx, X, c = gp.ctx, getX(gp), getc(gp)
    d, n = size(X)
    for j = 1:n
        xj = @view X[:,j]
        cj = c[j]
        kernel_gθ!(g, ctx, xj, xj, (invK[j,j] - cj*cj)/2)
        for i = j+1:n
            xi = @view X[:,i]
            ci = c[i]
            kernel_gθ!(g, ctx, xi, xj, (invK[i,j] - ci*cj))
        end
    end
    g
end

function nll_gθ(gp :: GPPContext, invK :: AbstractMatrix)
    nθ = nhypers(gp.ctx)
    nll_gθ!(zeros(nθ, gp, invK))
end

function nll_gθz!(g::AbstractVector, gp :: GPPContext, invK :: AbstractMatrix)
    ctx, X, c, s = gp.ctx, getX(gp), getc(gp), gp.η
    nll_gθ!(gp, gp, invK)
    g[nhypers(ctx)+1] = (tr(invK) - c'*c)*s/2
    g
end

function nll_gθz(gp :: GPPContext, invK)
    nll_gθz!(zeros(nhypers(gp.ctx) + 1), gp, invK)
end

nll_gθz(gp :: GPPContext) = nll_gθz(gp, getKC(gp)\I)

function whiten_matrix!(δK :: AbstractMatrix, KC :: Cholesky)
    ldiv!(KC.L, δK)
    rdiv!(δK, KC.U)
    δK
end

# TODO: not obvious why I'd need to whiten multiple matrices.

function mul_slices!(result, As, b)
    m, n, k = size(As)
    for j=1:k
        mul!(view(result,:,j), view(As,:,:,j), b)
    end
    result
end

# pack all hyperparameter matrices in a set (lengthscale, noise-variance).
function dθ_kernel!(δKs :: AbstractArray, ctx :: KernelContext,
                    X :: AbstractMatrix)
    n, n, d = size(δKs)
    for j = 1:n
        xj = @view X[:,j]
        δKjj = @view δKs[j,j,:]
        gθ_kernel!(δKjj, ctx, xj, xj)
        for i = j+1:n
            xi = @view X[:,i]
            δKij = @view δKs[i,j,:]
            δKji = @view δKs[j,i,:]
            gθ_kernel!(δKij, ctx, xi, xj)
            δKji[:] .= δKij
        end
    end
    δKs
end

function nll_Hθ(gp :: GPPContext)
    ctx, X, y, c, s = gp.ctx, getX(gp), gety(gp), getc(gp), gp.η
    d, n = size(X)
    nθ = nhypers(ctx)
    
    # Factorization and initial solves
    KC = getKC(gp)
    invK = KC\I
    c_tilde = KC.L\y
    φ = nll(gp)
    # z = log η
    nll_∂z = (tr(invK)-(c'*c))*s/2
    
    # Set up space for NLL, gradient, and Hessian (including wrt z)
    g = zeros(nθ+1)
    H = zeros(nθ+1,nθ+1)

    # Add Hessian contribution from kernel second derivatives
    d, n = size(X)
    for j = 1:n
        xj = @view X[:,j]
        cj = c[j]
        kernel_Hθ!(H, ctx, xj, xj, (invK[j,j]-cj*cj)/2)
        for i = j+1:n
            xi = @view X[:,i]
            ci = c[i]
            kernel_Hθ!(H, ctx, xi, xj, (invK[i,j]-ci*cj))
        end
    end
    H[nθ+1,nθ+1] = ∂z_nll

    # Set up whitened matrices δK̃ and products δK*c and δK̃*c̃
    δKs = zeros(n, n, nθ+1)
    kernel_dθ!(δKs, ctx, X)
    for j=1:n  δKs[j,j,nθ+1] = s  end
    δKtilde_s = whiten_matrices!(δKs, KC)
    δKtilde_c_tilde_s = mul_slices!(zeros(n,nθ+1), δKtilde_s, c_tilde)
    δKtilde_r = reshape(δK̃s, n*n, nθ+1)

    # Add Hessian contributions involving whitened matrices
    mul!(H, δKtilde_r', δKtilde_r, -0.5, 1.0)
    mul!(H, δKtilde_c_tilde_s', δKtilde_c_tilde_s, 1.0, 1.0)

    # And put together gradient
    for j=1:nθ
        g[j] = tr(view(δKtilde_s,:,:,j))/2
    end
    mul!(g, δKtilde_c_tilde_s', c_tilde, -0.5, 1.0)
    g[end] = ∂z_nll

    φ, g, H
end

@testset "Test NLL gradients" begin
    Zk, y = test_setup2d((x,y) -> x^2 + y)
    s, l = 1e-4, 1.0
    z=log(s)

    gp_SE_nll(l,z) = nll(GPPContext(KernelSE{2}(l), exp(z), Zk, y))
    g = nll_gθz(GPPContext(KernelSE{2}(l), s, Zk, y))

    @test g[1] ≈ diff_fd(l->gp_SE_nll(l,z), l) rtol=1e-6
    @test g[2] ≈ diff_fd(z->gp_SE_nll(l,z), z) rtol=1e-6
end

#@testset "Test NLL Hessians" begin
#    Zk, y = test_setup2d((x,y) -> x^2 + y)
#    s, ℓ = 1e-3, 0.89
#    z = log(s)
#
#    testf(ℓ,z) = Hθ_nll(GPPContext(KernelSE{2}(ℓ), exp(z), Zk, y))
#    ϕref, gref, Href = testf(ℓ, z)
#
#    @test gref[1] ≈ diff_fd(ℓ->testf(ℓ,z)[1][1], ℓ) rtol=1e-6
#    @test gref[2] ≈ diff_fd(z->testf(ℓ,z)[1][1], z) rtol=1e-6
#    @test Href[1,1] ≈ diff_fd(ℓ->testf(ℓ,z)[2][1], ℓ) rtol=1e-6
#    @test Href[1,2] ≈ diff_fd(ℓ->testf(ℓ,z)[2][2], ℓ) rtol=1e-6
#    @test Href[2,2] ≈ diff_fd(z->testf(ℓ,z)[2][2], z) rtol=1e-6
#    @test Href[1,2] ≈ Href[2,1]
#end

# ignoring the nll reduced version with scale factor ---too complicated.
function DψNLG0(z)
    φz = normpdf(z)
    Qz = normccdf(z)
    Gz = φz - z*Qz
    ψz = -log(Gz)
    dψz = Qz/Gz
    Hψz = (-φz*Gz + Qz^2)/Gz^2
    ψz, dψz, Hψz
end

function DψNLG2(z)
    # Approximate W by 20th convergent
    W = 0.0
    for k = 20:-1:1
        W = k/(z + W)
    end
    ψz = log1p(z/W) + 0.5*(z^2 + log(2π))
    dψz = 1/W
    Hψz = (1 - W*(z + W))/W^2
    ψz, dψz, Hψz
end

function logEI(z::Float64)
    DψNLG(z) = if z < 6.0 DψNLG0(z) else DψNLG2(z) end
end

function Hgx_αNLEI(gp :: GP, x :: AbstractVector, y_best :: Float64)
    Copt = getCopt(gp)
    μ, gμ, Hμ = mean(gp, x), mean_gx(gp, x), mean_Hx(gp, x)
    v, gv, Hv = Copt*var(gp, x), Copt*var_gx(gp, x), Copt*var_Hx(gp, x)

    σ = sqrt(v)
    gμs, Hμs = gμ/σ, Hμ/σ
    gvs, Hvs = gv/(2v), Hv/v
    
    u = (μ - y_best)/σ
    ψ, dψ, Hψ = logEI(u)

    α = -log(σ) + ψ
    dα = dψ*gμs - (1 + u*dψ)*gvs
    Hα = -0.5*(1.0 + u*dψ)*Hvs + dψ*Hμs + Hψ*gμs*gμs' +
         (2.0 + u^2*Hψ + 3.0*u*dψ)*gvs*gvs' +
         -(u*Hψ + dψ)*(gμs*gvs' + gvs*gμs')

    α, dα, Hα
end

# Test

#function optimize_EI(gp::GP, x_current :: AbstractVector, lo :: AbstractVector,
#        hi :: AbstractVector) 
#    y_best = minimum(gety(gp))
#    fun(x) = Hgx_αNLEI(gp, x, y_best)[1]
#    fun_g!(g, x) = copyto!(g, Hgx_αNLEI(gp, x, y_best)[2])
#    fun_H!(g, x) = copyto!(g, Hgx_αNLEI(gp, x, y_best)[3])
#    df = TwiceDifferentiable(fun, fun_g!, fun_H!, x0)
#    dfc = TwiceDifferentiableConstraints(lo, hi)
#    res = optimize(df, dfc, x0, IPNewton())
#end

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
