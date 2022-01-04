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
# Perhaps the following similarity transformation is a little nicer
S[:, 1] .= real.(W¹)
S[:, 2] .= real.(0.5 * (W² + W³))
S[:, 3] .= real.(0.5 * im * (W² - W³))
S[:, 4] .= real.(W⁴)

##
inv(S) * U * S
# Block Matrix Example
[S 0*I; 0*I S]

##
import StatisticalNonlocality: cheb, fourier_nodes, fourier_wavenumbers
N = 32
M = 32
D, z = cheb(M)
k = fourier_wavenumbers(N, L = 2)
x = fourier_nodes(M, a = -1, b = 1)

x = reshape(x, (N, 1))
z = reshape(z, (1, M + 1))

ψ¹ = sin.(x) .* sin.(z)
ψ² = cos.(x) .* sin.(z)
ψ³ = -sin.(x) .* sin.(z)
ψ⁴ = -cos.(x) .* sin.(z)

∂z = kron(I + zeros(N,N), D)
∂x = kron(D, I + zeros(M,M))

#=
# boundary indicies
∂Ωˣ = 
for i in ∂Ωˣ
    ∂x[j,:] .= 0.0
    ∂x[j,1] = 1.0
end
=#
qr(∂x) \ I