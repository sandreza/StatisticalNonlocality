using LinearAlgebra
import StatisticalNonlocality: ou_transition_matrix
import StatisticalNonlocality: fourier

filename = "three_state.jld2"

n = 2
M = ou_transition_matrix(n)

Λ, V = eigen(M)
Λ, W = eigen(M')

# Right eigenvectors
V³ = round.(V[:, 1] ./ V[1, 1])
V² = round.(V[:, 2] ./ V[1, 2])
V¹ = round.(V[:, 3] ./ V[1, 3])

# Left eigenvectors
W³ = round.(W[:, 1] ./ W[1, 1])
W² = round.(W[:, 2] ./ W[1, 2])
W¹ = round.(W[:, 3] ./ W[1, 3])

W[:, 1] .= W¹
W[:, 2] .= -W²
W[:, 3] .= W³

U = [-1 0 0; 0 0 0; 0 0 1] # advection operator structure
W * U * inv(W)

N = 128

D, x = fourier(N, a = 0, b = 2π)

γ = 1.0 # 1/ γ is the eddy timescale
U = 1.0
κ = 1.0

fluxkernel =
    -U * inv(κ .* D^2 - γ * I - U * D * inv(κ .* D^2 - 2 * γ * I) * U * D) * U
λ = eigvals(fluxkernel)
σ = svdvals(fluxkernel)


using JLD2
file = jldopen("data/" * filename, "a+")

file["kernel"] = fluxkernel
file["eigenvalues"] = λ
file["singularvalues"] = σ

close(file)
