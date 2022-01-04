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
        @test all(D * (x .^n ) - n * x .^(n-1) .≤ eps(N * 3.0))
    end
end

