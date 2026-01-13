using LinearAlgebra
using Random
using Optim
using StatsFuns

# TODO:
# 1. I would like to add conditionally positive definite kernels (spherical
# harmonics), which have less hyperparameters that we need to tune.
# 2. We would like to perform exact line search for the hyperparameter tuning.

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

function dist2(x :: AbstractVector{T}, y :: AbstractVector{T}) where {T}
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
updateθ(ctx :: KernelSE{d}, θ) where {d} = KernelSE{d}(θ[1])

kernel_func(ctx :: RBFKernelContext, x :: AbstractVector, y :: AbstractVector) =
    φ(ctx, dist(x,y)/ctx.l)

# convenience function.
function getθ(ctx :: KernelContext)
    θ = zeros(nhypers(ctx))
    getθ!(θ, ctx)
    θ
end

# TODO: implement spline interpolation here. This code should generate H.
#   TODO: spend some time writing exactly how this might go.
#   https://github.com/dnychka/fieldsRPackage/blob/main/fields/R/Tps.R

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

# Extend bordered matrix.
function kernel_lu!(AXX :: AbstractMatrix, ctx :: KernelContext, X ::
    AbstractMatrix, η :: Real)
    lu!(AXX)
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

#=
Julia docs
   Constructors: It is good practice to provide as few inner constructor methods
   as possible: only those taking all arguments explicitly and enforcing
   essential error checking and transformation. Additional convenience
   constructor methods, supplying default values or auxiliary transformations,
   should be provided as outer constructors that call the inner constructors to
   do the heavy lifting. This separation is typically quite natural.
=#

struct GPPContext{T <: KernelContext}
    ctx :: T
    n :: Integer # number of samples.
    p :: Integer # number of basis elements.
    η :: Float64
    Xstore  :: Matrix{Float64}
    Kstore  :: Matrix{Float64}
    Astore  :: Matrix{Float64} # bordered linear system.
    Hstore  :: Matrix{Float64}
    scratch :: Matrix{Float64}
    cstore  :: Vector{Float64}
    ystore  :: Vector{Float64}
end

getX(gp :: GPPContext) = view(gp.Xstore,:,1:gp.n)
getc(gp :: GPPContext) = view(gp.cstore,1:gp.n)
gety(gp :: GPPContext) = view(gp.ystore,1:gp.n)
getK(gp :: GPPContext) = view(gp.Kstore,1:gp.n,1:gp.n)
getKC(gp :: GPPContext) = Cholesky(UpperTriangular(getK(gp)))
getH(gp :: GPPContext) = view(gp.Hstore,1:gp.n,1:gp.p) # tall skinny
getA(gp :: GPPContext) = view(gp.Astore,1:gp.n+gp.p,1:gp.n+gp.p)
#getA_LU(gp :: GPPContext) = LU(getA(gp))
capacity(gp :: GPPContext) = length(gp.ystore)
getXrest(gp :: GPPContext) = view(gp.Xstore,:,gp.n+1:capacity(gp))
getyrest(gp :: GPPContext) = view(gp.ystore,gp.n+1:capacity(gp))
getXrest(gp :: GPPContext, m) = view(gp.Xstore,:,gp.n+1:gp.n+m)
getyrest(gp :: GPPContext, m) = view(gp.ystore,gp.n+1:gp.n+m)

function GPPContext(ctx :: KernelContext, p :: Integer, η :: Float64,
                    capacity :: Integer)
    d = ndims(ctx)
    Xstore  = zeros(d, capacity)
    Kstore  = zeros(capacity, capacity)
    Astore  = zeros(capacity+p, capacity+p)
    Hstore  = zeros(p, capacity)
    scratch = zeros(capacity,max(d+1,3))
    cstore  = zeros(capacity)
    ystore  = zeros(capacity)
#    GPPContext(ctx, η, Xstore, Kstore, cstore, ystore, scratch, 0)
    GPPContext(ctx, 0, p, η, 
               Xstore, Kstore, Astore, Hstore, scratch,
               cstore, ystore)
end

refactor_K!(gp :: GPPContext) =
    kernel_cholesky!(getK(gp), gp.ctx, getX(gp), gp.η)

# TODO
refactor_A!(gp :: GPPContext) = lu!(getA(gp))

resolve!(gp :: GPPContext) = ldiv!(getKC(gp), copyto!(getc(gp), gety(gp)))

function add_points_KC!(gp :: GPPContext, m :: Integer)
    n = gp.n + m
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

function add_points!(gp :: GPPContext, m :: Integer)
    n = gp.n + m
    if gp.n > capacity(gp)
        error("proposed points exceed GP context capacity.")
    end
    
    # Create new object (same storage)
    gpnew = GPPContext(gp.ctx, n, gp.p, gp.η, gp.Xstore, gp.Kstore, gp.Astore,
                       gp.Hstore, gp.scratch, gp.cstore, gp.ystore)

    #Refactor (if start from 0) or extend Cholesky (if partly done)
    display(stacktrace())
    if gp.n == 0
        refactor_K!(gpnew)
        refactor_A!(gpnew)
    else
        println("call this here")
        add_points_KC!(gp, m)
        refactor_A!(gp)
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

function GPPContext(ctx :: KernelContext, p :: Integer, η :: Float64,
                    X :: Matrix{Float64}, y :: Vector{Float64})
    d, n = size(X)
    @assert d == ndims(ctx) "Mismatch in dimensions of X and kernel."
    gp = GPPContext(ctx, p, η, n)
    copy!(gp.Xstore, X)
    copy!(gp.ystore, y)
    add_points!(gp, n)
end

function remove_points!(gp :: GPPContext, m)
    @assert m <= gp.n "Cannot remove $m > $(gp.n) points."
    gpnew = GPPContext(gp.ctx, gp.η, gp.Xstore, gp.Kstore, gp.Astore,
                       gp.cstore, gp.ystore, gp.scratch, gp.n-m, gp.p)
   resolve!(gpnew)
   gpnew
end

function change_kernel_nofactor!(gp :: GPPContext, ctx :: KernelContext,
                                 η :: Float64)
    GPPContext(ctx, η, gp.Xstore, gp.Kstore, gp.Astore,
               gp.cstore, gp.ystore, gp.scratch, gp.n, gp.p)
end

function change_kernel!(gp :: GPPContext, ctx :: KernelContext, η :: Float64)
    gpnew = change_kernel_nofactor!(gp, ctx, η)
    refactor_K!(gpnew)
    refactor_A!(gpnew)
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
    p = 1
    gp = GPPContext(ctx, p, 0.0, Zk, y)

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

# David: why is δ[logdet(A)] = δ[det(A)]/det(A)?

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
    nll_gθ!(zeros(nθ), gp, invK)
end

function nll_gθz!(g::AbstractVector, gp :: GPPContext, invK :: AbstractMatrix)
    ctx, c, s = gp.ctx, getc(gp), gp.η
    nll_gθ!(g, gp, invK)
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

function whiten_matrices!(δKs :: AbstractArray, KC :: Cholesky)
    n, n, m = size(δKs)
    for i = 1:m
        whiten_matrix!(view(δKs,:,:,i), KC)
    end
    δKs
end

function mul_slices!(result, As, b)
    m, n, k = size(As)
    for j=1:k
        mul!(view(result,:,j), view(As,:,:,j), b)
    end
    result
end

# pack all hyperparameter matrices in a set (lengthscale, noise-variance).
function kernel_dθ!(δKs :: AbstractArray, ctx :: KernelContext,
                    X :: AbstractMatrix)
    n, n, d = size(δKs)
    for j = 1:n
        xj = @view X[:,j]
        δKjj = @view δKs[j,j,:]
        kernel_gθ!(δKjj, ctx, xj, xj)
        for i = j+1:n
            xi = @view X[:,i]
            δKij = @view δKs[i,j,:]
            δKji = @view δKs[j,i,:]
            kernel_gθ!(δKij, ctx, xi, xj)
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
    # z is log η
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
    H[nθ+1,nθ+1] = nll_∂z

    # Set up whitened matrices δK̃ and products δK*c and δK̃*c̃
    δKs = zeros(n, n, nθ+1)
    kernel_dθ!(δKs, ctx, X)
    for j=1:n  δKs[j,j,nθ+1] = s  end
    δKtilde_s = whiten_matrices!(δKs, KC)
    δKtilde_c_tilde_s = mul_slices!(zeros(n,nθ+1), δKtilde_s, c_tilde)
    δKtilde_r = reshape(δKtilde_s, n*n, nθ+1)

    # Add Hessian contributions involving whitened matrices
    mul!(H, δKtilde_r', δKtilde_r, -0.5, 1.0)
    mul!(H, δKtilde_c_tilde_s', δKtilde_c_tilde_s, 1.0, 1.0)

    # And put together gradient
    for j=1:nθ
        g[j] = tr(view(δKtilde_s,:,:,j))/2
    end
    mul!(g, δKtilde_c_tilde_s', c_tilde, -0.5, 1.0)
    g[end] = nll_∂z

    φ, g, H
end

function monitor(ctx, η, φ, gφ)
    l = ctx.l
    println("($l, $η)\t$φ\t$(norm(gφ))")
#    push!(normgs, norm(gφ))
end

function newton_hypers0(gp :: GPPContext;
                        niters = 12, max_dz=3.0,
                        monitor = (ctx, η, φ, gφ)->nothing)
    ctx, η = gp.ctx, gp.η
    θ = getθ(ctx)
    for j = 1:niters
        φ, gφ, Hφ = nll_Hθ(gp)
        monitor(ctx, η, φ, gφ)
        u = -Hφ\gφ
        if abs(u[end]) > max_dz
            u *= max_dz/abs(u[2])
        end
        θ[:] .+= u[1:end-1]
        ctx = updateθ(ctx, θ)
        η *= exp(u[end])
        gp = change_kernel!(gp, ctx, η)
    end
    gp
end

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

DψNLG(z) = if z < 6.0 DψNLG0(z) else DψNLG2(z) end

function Hgx_αNLEI(gp :: GPPContext, x :: AbstractVector, y_best :: Float64)
    #Copt = getCopt(gp)
    μ, gμ, Hμ = mean(gp, x), mean_gx(gp, x), mean_Hx(gp, x)
    v, gv, Hv = var(gp, x), var_gx(gp, x), var_Hx(gp, x)

    σ = sqrt(v)
    gμs, Hμs = gμ/σ, Hμ/σ
    gvs, Hvs = gv/(2v), Hv/v
    
    u = (μ - y_best)/σ
    ψ, dψ, Hψ = DψNLG(u)

    α = -log(σ) + ψ
    dα = dψ*gμs - (1 + u*dψ)*gvs
    Hα = -0.5*(1.0 + u*dψ)*Hvs + dψ*Hμs + Hψ*gμs*gμs' +
         (2.0 + u^2*Hψ + 3.0*u*dψ)*gvs*gvs' +
         -(u*Hψ + dψ)*(gμs*gvs' + gvs*gμs')

    α, dα, Hα
end

function EI_optimize(gp :: GPPContext, x0 :: AbstractVector,
                     lo :: AbstractVector, hi :: AbstractVector)
    fopt = minimum(gety(gp))
    fun(x) = Hgx_αNLEI(gp, x, fopt)[1]
    fun_grad!(g, x) = copyto!(g, Hgx_αNLEI(gp, x, fopt)[2])
    fun_hess!(H, x) = copyto!(H, Hgx_αNLEI(gp, x, fopt)[3])
    df = TwiceDifferentiable(fun, fun_grad!, fun_hess!, x0)
    dfc = TwiceDifferentiableConstraints(lo, hi)
    res = optimize(df, dfc, x0, IPNewton())
end

function EI_optimize(gp :: GPPContext,
                     lo :: AbstractVector, hi :: AbstractVector;
                     nstarts = 10, verbose=true)
    bestα = Inf
    bestx = [0.0; 0.0]
    for j = 1:10
        z = lo + (hi-lo).*rand(length(lo))
        res_ei = EI_optimize(gp, z, [0.0; 0.0], [1.0; 1.0])
        if verbose
            println("From $z: $(Optim.minimum(res_ei)) at \
            $(Optim.minimizer(res_ei))")
        end

        if Optim.minimum(res_ei) < bestα
            bestα = Optim.minimum(res_ei)
            bestx[:] = Optim.minimizer(res_ei)
        end
    end
    bestα, bestx
end

let
    testf(x,y) = x^2+y
    Zk, y = test_setup2d(testf)
    gp = GPPContext(KernelSE{2}(0.2), 1, 1e-7, 20)
    gp = add_points!(gp, Zk, y)
    println("updated A ")
    display(getA(gp))
#    for k = 1:5
#        bestα, bestx = EI_optimize(gp, [0.0; 0.0], [1.0; 1.0], verbose=false)
#        y = testf(bestx ... )
#        gp = add_point!(gp, bestx, y)
#        println("updated A ", getA_LU(gp))
#        gp = newton_hypers0(gp, monitor=monitor)
#        println("$k: EI=$(exp(-bestα)), f($bestx) = $y")
#    end
#    println("Best found: $(minimum(gety(gp)))")
end
