import StatisticalNonlocality: ou_transition_matrix, uniform_phase
import StatisticalNonlocality: discrete_laplacian

@testset "Transition Matrix Column Sum Zero" begin
    for n in 1:10
        M = ou_transition_matrix(n)
        @test all(abs.(sum(M, dims = 1)) .≤ eps(n / 2))
        if n > 1
            M = uniform_phase(n)
            @test all(abs.(sum(M, dims = 1)) .≤ eps(n / 2))
            Δ = discrete_laplacian(n)
            @test all(abs.(sum(Δ, dims = 1)) .≤ eps(n^2 / 2))
        end
    end
end

@testset "OU Transition Matrix Correctness Test for n in 1:2" begin
    n = 1
    M = ou_transition_matrix(n)
    Mh = [
        -1/2 1/2
        1/2 -1/2
    ]
    @test all(abs.(Mh - M) .≤ eps(n / 2))

    n = 2
    M = ou_transition_matrix(n)
    Mh = [
        -1 1/2 0
        1 -1 1
        0 1/2 -1
    ]
    @test all(abs.(Mh - M) .≤ eps(n / 2))
end

@testset "Uniform Phase Transition Matrix Correctness Test for n = 4" begin
    n = 4
    M = uniform_phase(n)
    Mh = [-1 1 0 0; 0 -1 1 0; 0 0 -1 1; 1 0 0 -1]
    @test all(abs.(Mh - M) .≤ eps(n / 2))

end

@testset "Eigenvalue Test for n in 1:10" begin
    for n in 1:10
        λ_exact = collect((-n):0)
        λ_approx = eigvals(ou_transition_matrix(n))
        @test all(abs.(λ_exact - λ_approx) .≤ eps(20.0 * n))
    end
end

@testset "Symmetry Test for Discrete Laplacian" begin
    for n in 4:10
        Δ = discrete_laplacian(n)
        @test norm(Δ - Δ') ≤ eps(norm(Δ))
    end
end