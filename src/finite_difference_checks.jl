using LinearAlgebra
using Random
using Distributions

# check log EI gradient
# 0. gradient of k(x,x') = -sigma^2 exp(-0.5l^{-2} ||x - x'||^2) INCORRECT!
# 1. gradient of mu(x)
# 2. gradient of sigma(x)

# gradient of k(x,x')
N = 10
X = randn(N)

function rbf(x)
    x0 = 0.0
    lengthscale = 0.5
    variance = 1.0
    return variance*exp(-0.5*sum((x0 .- x).^2)/(lengthscale^2))
end

function grad_k(k, x)
    lengthscale = 0.5
    return k(x)*(-1/(lengthscale^2)*x)
end

# I want to be able to compare functions with whatever kind of output.
# we check approximation by comparing norms.
function approx_check(xref, x; rtol=1e-6, atol=0.0)
    # don't compute relative error up front because what if xref is zero?
    abs_err = norm(xref - x)
    ref_norm = norm(xref)
    if abs_err > rtol*ref_norm + atol
        error("Check failed: $abs_err > $rtol * $ref_norm+ $atol")
    end
    return abs_err / ref_norm
end

# a finite difference happens in one dimension.
fd(f, x; h=1e-6) = (f(x+h) - f(x-h))/(2h)

fd_check(df_ref, f, x; h=1e-6, rtol=1e-6, atol=0.0) = approx_check(df_ref, fd(f, x, h=h))
# OR
# TODO: get this way to work too.
#fd_check(df_ref, f, x; h=1e-6, rtol=1e-6, atol=0.0) = approx_check(df_ref, x -> (f(x+h) -
#                                                                   f(x-h))/(2h))
let s = 0.89
    print(fd_check(grad_k(rbf, s), rbf, s))
end
