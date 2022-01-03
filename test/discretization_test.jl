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

@testset "Chebyshev Grid and Wavenumbers" begin
    N = 2
    D, x = cheb(N)
    xᴬ = [1, 0, -1]
    Dᴬ = [3/2 -2 1/2; 1/2 0 1/2; -1/2 2.0 -3/2]
    @test all(x - xᴬ .≤ eps(N * 1.0))
    @test all(D .- Dᴬ .≤ eps(N * 1.0))
end

