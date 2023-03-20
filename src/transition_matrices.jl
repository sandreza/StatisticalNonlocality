# Discrete OU process ALA Charlie Doering
function ou_transition_matrix(n)
    Mⱼₖ = zeros(n + 1, n + 1)
    δ(j, k) = (j == k) ? 1 : 0

    for j in 0:n, k in 0:n
        jj = j + 1
        kk = k + 1
        Mⱼₖ[jj, kk] =
            (-n * δ(j, k) + k * δ(j + 1, k) + (n - k) * δ(j - 1, k)) / 2
    end
    return Mⱼₖ
end

# Discrete Phase Transition ala Glenn Flierl
function uniform_phase(N)
    @assert N > 1
    T = zeros(N, N)
    for i in 1:N
        T[i, i] = -1.0
        T[i, i%N+1] = 1.0
    end
    return T
end

# Discrete Laplacian
function discrete_laplacian(n)
    Δ = zeros(n, n)
    for i in 2:n-1
        Δ[i, i] = -2
        Δ[i, i+1] = 1
        Δ[i, i-1] = 1
    end
    Δ[1, 1] = Δ[end, end] = -1
    Δ[1, 2] = 1
    Δ[end, end-1] = 1
    return Δ
end


function advection_matrix_central(N; Δx=1.0, u=1.0)
    A = zeros(N, N)
    for i in 1:N
        A[(i-1)%N+1, (i-1*0)%N+1] = 1
        A[(i+1)%N+1, (i-1*0)%N+1] = -1
    end
    return u * A / 2Δx
end

function advection_matrix_upwind(N; Δx=1.0, u=1.0)
    A = zeros(N, N)
    for i in 1:N
        A[(i-1)%N+1, (i-1*0)%N+1] = (u + abs(u)) / 2
        A[i, i] = -u
        A[(i+1)%N+1, (i-1*0)%N+1] = (u - abs(u)) / 2
    end
    return A / Δx
end

function discrete_laplacian_periodic(N; Δx=1.0, κ=1.0)
    Δ = zeros(N, N)
    for i in 1:N
        Δ[i, i] = -2
        Δ[(i-1)%N+1, (i-1*0)%N+1] = 1
        Δ[(i+1)%N+1, (i-1*0)%N+1] = 1
    end
    return κ * Δ / (Δx^2)
end