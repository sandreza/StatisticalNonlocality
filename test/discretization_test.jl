import StatisticalNonlocality: cheb, fourier_nodes, fourier_wavenumbers

@testset "Fourier Grid and Wavenumbers" begin
    for N in 1:5
        x = fourier_nodes(N, a = 0, b = N)
        @test all(x - collect(0:N-1) .≤ eps(N * 1.0))
    end

    N = 4
    k = fourier_wavenumbers(N, L = 2π)
    @test all(k - [0, 1, -2, -1] .≤ eps(N * 1.0))

end

@testset "Fourier Derivative" begin
    N = 8
    a, b = 0, 2π
    k = fourier_wavenumbers(N, L = b - a)
    x = fourier_nodes(N, a = a, b = b)

    ℱ = fft(I + zeros(N, N), 1)
    ℱ⁻¹ = ifft(I + zeros(N, N), 1)
    # build differentiation matrix in real space
    Dx = real.(ℱ⁻¹ * Diagonal(im .* k) * ℱ)
    # only wavenumbers 0, 1, 2, ..., div(N,2)-1 are well represented
    # differentiation matrix is exact for n = 0, sin(x), cos(x), sin(2x), cos(2x)
    # all the wave up to div(N,2)-1
    for n in 1:div(N, 2)-1
        @test all(abs.(Dx * sin.(n * x) - n * cos.(n * x)) .≤ eps(10.0 * N))
        @test all(abs.(Dx * cos.(n * x) + n * sin.(n * x)) .≤ eps(10.0 * N))
    end
end

@testset "1D Helmholtz Equation" begin
    N = 8
    a, b = 0, 2π
    k = fourier_wavenumbers(N, L = b - a)
    x = fourier_nodes(N, a = a, b = b)
    ℱ = fft(I + zeros(N, N), 1)
    ℱ⁻¹ = ifft(I + zeros(N, N), 1)
    # build differentiation matrix in real space
    Dx = real.(ℱ⁻¹ * Diagonal(im .* k) * ℱ)
    H = Dx * Dx - I
    rhs = sin.(x) + sin.(k[3] * x)
    numerical = H \ rhs
    analytic = sin.(x) / (-1 - k[2]^2) + sin.(k[3] * x) / (-1 - k[3]^2)
    tolerance = 1e2 * eps(maximum(analytic))
    @test norm(numerical - analytic, Inf) ≤ tolerance
end

@testset "Chebyshev Grid and Wavenumbers" begin
    N = 2
    D, x = cheb(N)
    xᴬ = [1, 0, -1]
    Dᴬ = [3/2 -2 1/2; 1/2 0 1/2; -1/2 2.0 -3/2]
    @test all(x - xᴬ .≤ eps(N * 1.0))
    @test all(D .- Dᴬ .≤ eps(N * 1.0))
end

@testset "Chebyshev Derivative" begin
    N = 8
    D, x = cheb(N)
    for n in 3:N-1
        @test all(D * (x .^ n) - n * x .^ (n - 1) .≤ eps(N * 3.0))
    end
end

@testset "Chebyshev Poisson" begin
    N = 8
    D, x = cheb(N)

    Δ = D * D
    Δ[1, :] .= 0.0
    Δ[1, 1] = 1.0
    Δ[end, :] .= 0.0
    Δ[end, end] = 1.0

    rhs = ones(N + 1) * 2
    rhs[1] = 0.0
    rhs[end] = 0.0
    numerical = Δ \ rhs
    analytic = x .^ 2 .- 1
    tolerance = 1e2 * eps(norm(analytic, Inf))
    @test norm(numerical - analytic, Inf) ≤ tolerance
end



@testset "Fourier-Chebyshev Heat Equation" begin
    N = 8
    M = 8
    Dz, z = cheb(M)
    a, b = 0, 2π
    k = fourier_wavenumbers(N, L = b - a)
    x = fourier_nodes(N, a = a, b = b)

    ℱ = fft(I + zeros(N, N), 1)
    ℱ⁻¹ = ifft(I + zeros(N, N), 1)
    Dx = real.(ℱ⁻¹ * Diagonal(im .* k) * ℱ)
    x = reshape(x, (N, 1))
    z = reshape(z, (1, M + 1))

    ∂z = kron(Dz, I + zeros(N, N))
    ∂x = kron(I + zeros(M + 1, M + 1), Dx)

    Δ = ∂x * ∂x + ∂z * ∂z
    bΔ = copy(Δ)
    # Figure out boundary indices without thinking
    zlifted = sparse(kron(Diagonal(z[:]), I + zeros(N, N)))
    ∂Ω¹ = zlifted.rowval[zlifted.nzval.==z[1]]
    ∂Ω² = zlifted.rowval[zlifted.nzval.==z[end]]
    ∂Ω = vcat(∂Ω¹, ∂Ω²)

    # apply Dirichlet boundary conditions
    for ∂i in ∂Ω
        bΔ[∂i, :] .= 0.0
        bΔ[∂i, ∂i] = 1.0
    end

    # Simple Test: Δϕ = 1 & ϕ(boundary) = 1 => z^2 /2 + 1/2
    rhs = ones(size(Δ)[2])
    numerical = bΔ \ rhs
    analytic = @. x * 0 + z^2 / 2 + 1 / 2
    numerical = reshape(numerical, size(analytic))
    e1 = norm(analytic[:] - numerical[:], Inf)
    tolerance = 1e2 * eps(norm(analytic[:], Inf))
    @test e1 < tolerance

    # Simple Test 2: Δϕ = 2 * sin(2 * x) + (z^2 - 1) * 4 * sin(2x)
    # & ϕ(boundary) = 0 
    # => (z^2 - 1) * sin(2 x)
    rhs = @. 2 * sin(k[3] * x) - (z^2 - 1) * k[3]^2 * sin(k[3] * x)
    rhs = rhs[:] # flatten rhs
    # homogenous boundary conditions
    for ∂i in ∂Ω
        rhs[∂i] = 0.0
    end
    numerical = bΔ \ rhs
    analytic = @. (z^2 - 1) * sin(k[3] * x)
    numerical = reshape(numerical, size(analytic))
    e2 = norm(analytic[:] - numerical[:], Inf)
    tolerance = 1e2 * eps(norm(analytic[:], Inf))
    @test e2 < tolerance

    # Simple Test 3: A block diagonal equation of 2: [Δ 0 ; 0 Δ] 
    Δ = ∂x^2 + ∂z^2
    bΔ = [Δ 0*Δ; 0*Δ Δ]

    zlifted = sparse(kron(Diagonal(z[:]), I + zeros(N, N)))
    zlifted = [zlifted 0*zlifted; 0*zlifted zlifted]
    ∂Ω¹ = zlifted.rowval[zlifted.nzval.==z[1]]
    ∂Ω² = zlifted.rowval[zlifted.nzval.==z[end]]
    ∂Ω = vcat(∂Ω¹, ∂Ω²)


    rhs1 = @. 2 * sin(k[3] * x) - (z^2 - 1) * k[3]^2 * sin(k[3] * x)
    rhs2 = @. 2 * sin(k[4] * x) - (z^2 - 1) * k[4]^2 * sin(k[4] * x)
    rhs = vcat(rhs1[:], rhs2[:]) # flatten rhs
    # homogenous boundary conditions
    for ∂i in ∂Ω
        bΔ[∂i, :] .= 0.0
        bΔ[∂i, ∂i] = 1.0
        rhs[∂i] = 0.0
    end
    numerical = bΔ \ rhs
    analytic1 = @. (z^2 - 1) * sin(k[3] * x)
    analytic2 = @. (z^2 - 1) * sin(k[4] * x)
    analytic = vcat(analytic1[:], analytic2[:])
    e3 = norm(analytic - numerical, Inf)
    tolerance = 1e2 * eps(norm(analytic[:], Inf))
    @test e3 < tolerance

end