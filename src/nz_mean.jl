using LinearAlgebra
using Distributions
using StatsFuns
using Random
using Plots
using Optim
using DispatchDoctor: @stable

function kronecker_quasirand(d :: Int32, N :: Int32, start=0)
    φ = 1.0 + 1.0/d
    for k = 1:10
        gφ = φ^(d+1) - φ - 1
        dgφ = (d+1)*φ^d - 1
        φ -= gφ/dgφ
    end
    αs = [mod(1.0/φ^j, 1.0) for j=1:d]

    # Compute the quasi-random sequence.
    Z = zeros(d, N)
    for j = 1:N
        for i = 1:d
            Z[i,j] = mod(0.5 + (start+j)*αs[i], 1.0)
        end
    end
    Z
end

# squared exponential kernel.
function Dφ_SE(s :: Float64)
    φ = exp(-s^2/2)
    dφ_div = -φ
    dφ = dφ_div*s
    Hφ = (-1 + s^2)*φ
    φ, dφ_div, dφ, Hφ
end

# Question for David: is there any drawback to using @stable for every function call?
@stable function dist2(x :: AbstractVector{T}, y :: AbstractVector{T}) where {T}
    s = zero(T)
    for k = 1:length(x)
        dk = x[k] - y[k]
        s += dk * dk
    end
    s
end

# fast distance calculator; norm(x-y) creates an intermediate vector.
@stable dist(x :: AbstractVector{T}, y :: AbstractVector{T}) where {T} = sqrt(dist2(x,y))

abstract type KernelContext end
# Question for David: what is this? Custom pretty-printing?
(ctx :: KernelContext)(args ...) = kernel(ctx, args ... )

abstract type RBFKernelContext{d} <: KernelContext end

ndims(::RBFKernelContext{d}) where {d} = d

function getθ(ctx :: KernelContext)
    θ = zeros(nhypers(ctx))
    getθ!(θ, ctx)
    θ
end

macro rbf_simple_kernel(T, φ_rbf, Dφ_rbf)
    T, φ_rbf, Dφ_rbf = esc(T), esc(φ_rbf), esc(Dφ_rbf)
    quote
        struct $T{d} <: $(esc(:RBFKernelContext)){d}
            l :: Float64
        end
        $(esc(:φ))(::$T, s) = $φ_rbf(s)
        $(esc(:Dφ))(::$T, s) = $Dφ_rbf(s)
        $(esc(:nhypers))(::$T) = 1
        $(esc(:getθ!))(θ, ctx :: $T) = θ[1]=ctx.l
        $(esc(:updateθ))(ctx ::$T{d}, θ) where {d} = $T{d}(θ[1])
    end
end

@rbf_simple_kernel(KernelSE, φ_SE, Dφ_SE)

kernel(ctx :: RBFKernelContext, x :: AbstractVector, y :: AbstractVector) =
    φ(ctx, dist(x, y)/ctx.l)
