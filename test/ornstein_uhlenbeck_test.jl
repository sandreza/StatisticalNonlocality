
import StatisticalNonlocality: ou_transition_matrix

@testset "Transition Matrix Column Sum Zero" begin
    for n in 1:10
        M = ou_transition_matrix(n)
        @test all(abs.(sum(M, dims = 1)) .≤ eps(n / 2))
    end
end

@testset "Transition Matrix Correctness Test for n in 1:2" begin
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

@testset "Eigenvalue Test for n in 1:10" begin
    for n in 1:10
        λ_exact = collect((-n):0)
        λ_approx = eigvals(ou_transition_matrix(n))
        @test all(abs.(λ_exact - λ_approx) .≤ eps(20.0 * n))
    end
end
