using LinearAlgebra
T = [-1 1 0 0; 0 -1 1 0; 0 0 -1 1; 1 0 0 -1]

Λ, V = eigen(T)
Λ, W = eigen(T')

# Right eigenvectors
V⁴ = round.(V[:, 1] ./ V[1, 1])
V³ = round.(V[:, 2] ./ V[1, 2])
V² = round.(V[:, 3] ./ V[1, 3])
V¹ = round.(V[:, 4] ./ V[1, 4])

# Left eigenvectors
W⁴ = round.(W[:, 1] ./ W[1, 1])
W³ = round.(W[:, 2] ./ W[1, 2])
W² = round.(W[:, 3] ./ W[1, 3])
W¹ = round.(W[:, 4] ./ W[1, 4])

# Flip Order (take into account later)
W[:, 1] .= W¹
W[:, 2] .= W²
W[:, 3] .= W³
W[:, 4] .= W⁴
inv(W) * 4

# Quick Check (right eigenvector)
T * V - V * Diagonal(Λ)
maximum(abs.(T * V - V * Diagonal(Λ))) ≤ eps(10.0)
# Quick Check (left eigenvector)
T' * W - W * Diagonal(reverse(Λ))    # since we flipped the order so that index 1 corresponds to the lowest eigenvalue
W' * T - Diagonal(reverse(Λ))' * W'  # same thing, recall that ' means adjoint, thus takes the conjucate
maximum(abs.(T' * W - W * Diagonal(reverse(Λ)))) ≤ eps(10.0)

U = [
    exp(im * 0) 0 0 0
    0 exp(im * π / 2) 0 0
    0 0 exp(im * π) 0
    0 0 0 exp(im * 3π / 2)
] # advection operator structure
W * U * inv(W)

##
