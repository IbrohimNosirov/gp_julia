using Test
include("zero_mean_gp.jl")

#@testset "Compare kernel matrix constructors" begin
#    Zk = kronecker_quasirand(2,10)
#    k_ctx = KernelSE{2}(1.0)
#
#    # Comprehension-based eval
#    KXX1 = [k_ctx(x,y) for x in eachcol(Zk), y in eachcol(Zk)]
#    KXz1 = [k_ctx(x,Zk[:,1]) for x in eachcol(Zk)]
#
#    # Dispatch through call mechanism
#    KXX2 = kernel_alloc(k_ctx, Zk)
#    KXX3 = kernel_alloc(k_ctx, Zk, Zk)
#    KXz2 = kernel_alloc(k_ctx, Zk, Zk[:,1])
#
#    @test KXX1 ≈ KXX2
#    @test KXX1 ≈ KXX3
#    @test KXz1 ≈ KXX1[:,1]
#    @test KXz2 ≈ KXX1[:,1]
#end
#
#let 
#    Zk = kronecker_quasirand(2, 10)
#    k_ctx = KernelSE{2}(1.0)
#    Ktemp = zeros(10,10)
#    Kvtemp = zeros(10)
#    Zk1 = Zk[:,1]
#
#    KXX1 = @time kernel_alloc(k_ctx, Zk)
#    KXX2 = @time kernel_alloc(k_ctx, Zk, Zk)
#    KXz2 = @time kernel_alloc(k_ctx, Zk, Zk[:,1])
#    @time kernel!(Ktemp, k_ctx, Zk)
#    @time kernel!(Ktemp, k_ctx, Zk, Zk)
#    @time kernel!(Kvtemp, k_ctx, Zk, Zk1)
#end
#
#let
#    function fd_check_Dφ(Dφ, s; kwargs ... )
#        φ, dφ_div, dφ, Hφ = Dφ(s; kwargs ... )
#        @test dφ_div*s ≈ dφ
#        @test dφ ≈ diff_fd(s->Dφ(s; kwargs ... )[1], s) rtol=1e-6
#        @test Hφ ≈ diff_fd(s->Dφ(s; kwargs ... )[3], s) rtol=1e-6
#    end
#
#    @testset "SE kernel derivative check" begin
#        s = .123
#        @testset "SE" fd_check_Dφ(Dφ_SE, s)
#    end
#
#    @testset "Kernel hyper derivatives" begin
#        x = [0.1; 0.2]
#        y = [0.8; 0.8]
#        l = 0.123
#        k_ctx(l) = KernelSE{2}(l)
#        # convenience function to be able to call struct as a function
#        k(l) = kernel_func(k_ctx(l), x, y)
#        g(l) = kernel_gθ_alloc(k_ctx(l), x, y)[1]
#        H(l) = kernel_Hθ_alloc(k_ctx(l), x, y)[1,1]
#        @test g(l) ≈ diff_fd(k, l) rtol=1e-6
#        @test H(l) ≈ diff_fd(g, l) rtol=1e-6
#    end
#
#    @testset "Kernel spatial derivatives" begin
#        x, y, dx  = [0.2; 0.4], [0.3; 0.7], [0.3; 0.5]
#        l = 0.32
#        k_ctx = KernelSE{2}(l)
#        # convenience function to be able to call struct as a function
#        k(x) = kernel_func(k_ctx, x, y)
#        g(x) = kernel_gx_alloc(k_ctx, x, y)
#        H(x) = kernel_Hx_alloc(k_ctx, x, y)
#        @test g(x)' * dx ≈ diff_fd(s->k(x + s*dx)) rtol=1e-6 
#        @test H(x) * dx ≈ diff_fd(s->g(x + s*dx)) rtol=1e-6
#    end
#end
#
#let
#    Zk = kronecker_quasirand(2, 10)
#    k_ctx = KernelSE{2}(1.0)
#    Ktemp = zeros(10,10)
#    Kvtemp = zeros(10)
#    Zk1 = Zk[:,1]
#    η = 1e-8
#
#    KXX = @time kernel_cholesky(k_ctx, Zk)
#    @time kernel_cholesky!(Ktemp, k_ctx, Zk)
#    KXX2 = @time kernel_cholesky(k_ctx, Zk, η)
#    @time kernel_cholesky!(Ktemp, k_ctx, Zk, η)
#end
#
#@testset "Check Cholesky extension" begin
#    # Pre-generated random matrix
#    A = [1.362     0.767029  0.991061  1.07994  1.35389;
#         0.767029  1.55874   1.36436   1.5897   1.34834;
#         0.991061  1.36436   1.49529   1.46581  1.43195;
#         1.07994   1.5897    1.46581   1.97641  1.69469;
#         1.35389   1.34834   1.43195   1.69469  2.19448]
#    A_full = cholesky(A)
#
#    Chol_1 = extend_cholesky!(A, 0, 3)
#    @test Chol_1.U ≈ A_full.U[1:3,1:3]
#    Chol_2 = extend_cholesky!(A, 3, 5)
#    @test Chol_2.U ≈ A_full.U
#end
#
#let
#    # Set up sample points and test function
#    testf(x,y) = x^2 + y
#    Zk, y = test_setup2d(testf)
#    ctx = KernelSE{2}(1.0)
#
#    # Form kernel Cholesky and weights
#    KC = kernel_cholesky(ctx, Zk)
#    c = KC\y
#
#    # Evaluate true function and GP at a test point
#    z = [0.456; 0.436]
#    fz = testf(z...)
#    μz, σz = eval_GP(KC, ctx, Zk, c, z)
#
#    # Compare GP to true function
#    zscore = (fz-μz)/σz
#    println("""
#        True value:       $fz
#        Posterior mean:   $μz
#        Posterior stddev: $σz
#        z-score:          $zscore
#        """)
#end
#
#@testset "Update points and kernel in GPPContext" begin
#    testf(x,y) = x^2+y
#    Zk, y = test_setup2d(testf)
#    ctx = KernelSE{2}(1.0)
#    ctx2 = KernelSE{2}(0.8)
#
#    gp1 = GPPContext(ctx, 0.0, Zk, y)
#    gpt = GPPContext(ctx, 0.0, 10)
#    gpt = add_points!(gpt, Zk, y)
#    @test getc(gpt) ≈ getc(gp1)
#
#    gpt = remove_points!(gpt, 2)
#    gp3 = GPPContext(ctx, 0.0, Zk[:,1:end-2], y[1:end-2])
#    @test getc(gpt) ≈ getc(gp3)
#
#    gpt = add_points!(gpt, Zk[:,end-1:end], y[end-1:end])
#    @test getc(gpt) ≈ getc(gp1)
#end
#
#@testset "Predictive mean and variance derivatives" begin
#    Zk, y = test_setup2d((x,y) -> x^2 + cos(3*y))
#    z, dz = [0.47; 0.47], [0.132; 0.0253]
#    gp = GPPContext(KernelSE{2}(0.5), 1e-8, Zk, y)
#    @test mean_gx(gp,z)'*dz ≈ diff_fd(s->mean(gp,z+s*dz))    rtol=1e-6
#    @test mean_Hx(gp,z)*dz  ≈ diff_fd(s->mean_gx(gp,z+s*dz)) rtol=1e-6
#    @test var_gx(gp,z)'*dz  ≈ diff_fd(s->var(gp,z+s*dz))     rtol=1e-6
#    @test var_Hx(gp,z)*dz   ≈ diff_fd(s->var_gx(gp,z+s*dz))  rtol=1e-6
#end
#
#@testset "Test gradients " begin
#    # David: is a second variational just a different direction vector?
#    A0, δA, ΔA, ΔδA = randn(10,10), rand(10,10), rand(10,10), rand(10,10)
#
#    δinv(A,δA) = -A\δA/A # ((A') \ (-A\δA)')' rdiv!(-A\δA,A) doesn't work? David
#    invAδA, invAΔA, invAΔδA = A0\δA, A0\ΔA, A0\ΔδA
#    ΔδinvA = (invAδA*invAΔA + invAΔA*invAδA - invAΔδA)/A0
#
#    @test δinv(A0,δA) ≈ diff_fd(s->inv(A0+s*δA)) rtol=1e-6
#    @test ΔδinvA ≈ diff_fd(s->δinv(A0+s*ΔA, δA+s*ΔδA)) rtol=1e-6
#
#    V = randn(10,10)
#    A = V*Diagonal(1.0.+rand(10))/V
#    δA, ΔA, ΔδA = randn(10,10), randn(10,10), randn(10,10)
#
#    δlogdet(A, δA) = tr(A\δA)
#    Δδlogdet(A, δA, ΔA, ΔδA) = tr(A\ΔδA)-tr((A\ΔA)*(A\δA))
#
#    @test δlogdet(A,δA) ≈ dot(inv(A'), δA)
#    @test δlogdet(A,δA) ≈ diff_fd(s->log(det(A+s*δA))) rtol=1e-6
#    @test Δδlogdet(A,δA,ΔA,ΔδA) ≈ diff_fd(s->δlogdet(A+s*ΔA,δA+s*ΔδA)) rtol=1e-6
#end
#
#@testset "Test NLL gradients" begin
#    Zk, y = test_setup2d((x,y) -> x^2 + y)
#    s, l = 1e-4, 1.0
#    z=log(s)
#
#    gp_SE_nll(l,z) = nll(GPPContext(KernelSE{2}(l), exp(z), Zk, y))
#    g = nll_gθz(GPPContext(KernelSE{2}(l), s, Zk, y))
#
#    @test g[1] ≈ diff_fd(l->gp_SE_nll(l,z), l) rtol=1e-6
#    @test g[2] ≈ diff_fd(z->gp_SE_nll(l,z), z) rtol=1e-6
#end
#
#@testset "Test NLL Hessians" begin
#    Zk, y = test_setup2d((x,y) -> x^2 + y)
#    s, l = 1e-3, 0.89
#    z = log(s)
#
#    testf(l,z) = nll_Hθ(GPPContext(KernelSE{2}(l), exp(z), Zk, y))
#    ϕref, gref, Href = testf(l, z)
#
#    @test gref[1] ≈ diff_fd(l->testf(l,z)[1][1], l) rtol=1e-6
#    @test gref[2] ≈ diff_fd(z->testf(l,z)[1][1], z) rtol=1e-6
#    @test Href[1,1] ≈ diff_fd(l->testf(l,z)[2][1], l) rtol=1e-6
#    @test Href[1,2] ≈ diff_fd(l->testf(l,z)[2][2], l) rtol=1e-6
#    @test Href[2,2] ≈ diff_fd(z->testf(l,z)[2][2], z) rtol=1e-6
#    @test Href[1,2] ≈ Href[2,1]
#end
#
## Test
#@testset "Test DψNLG0" begin
#    z = 0.123
#    @test DψNLG0(z)[2] ≈ diff_fd(z->DψNLG0(z)[1], z) rtol=1e-6
#    @test DψNLG0(z)[3] ≈ diff_fd(z->DψNLG0(z)[2], z) rtol=1e-6    
#end
#
#@testset "Test DψNLG2" begin
#    z = 5.23
#    @test DψNLG0(z)[1] ≈ DψNLG2(z)[1]
#    @test DψNLG0(z)[2] ≈ DψNLG2(z)[2]
#    @test DψNLG0(z)[3] ≈ DψNLG2(z)[3]    
#end
#
#@testset begin
#    function check_derivs3_NLEI(f)
#        Zk, y = test_setup2d((x,y)->x^2+y)
#        gp = GPPContext(KernelSE{2}(0.5), 1e-8, Zk, y)
#        z = [0.47; 0.47]
#        dz = randn(2)
#        fopt = -0.1
#        g(s)  = f(gp, z+s*dz, fopt)[1]
#        dg(s) = f(gp, z+s*dz, fopt)[2]
#        Hg(s) = f(gp, z+s*dz, fopt)[3]
#        @test dg(0)'*dz ≈ diff_fd(g) rtol=1e-6
#        @test Hg(0)*dz ≈ diff_fd(dg) rtol=1e-6
#    end
#    @testset "Test Hgx_αNLEI" check_derivs3_NLEI(Hgx_αNLEI)
#end

let
    testf(x,y) = x^2+y
    Zk, y = test_setup2d(testf)
    gp = GPPContext(KernelSE{2}(0.8), 1e-8, 20)
    gp = add_points!(gp, Zk, y)
    for k = 1:5
        bestα, bestx = EI_optimize(gp, [0.0; 0.0], [1.0; 1.0], verbose=false)
        y = testf(bestx...)
        gp = add_point!(gp, bestx, y)
        println("$k: EI=$(exp(-bestα)), f($bestx) = $y")
    end
    println("--- End loop ---")
    println("Best found: $(minimum(gety(gp)))")
end
