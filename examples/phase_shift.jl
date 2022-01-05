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
using StatisticalNonlocality, LinearAlgebra, FFTW, SparseArrays
import StatisticalNonlocality: cheb, fourier_nodes, fourier_wavenumbers
import StatisticalNonlocality: droprelativezeros!
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

ψ¹ = sin.(x) .* sin.(z)
ψ² = cos.(x) .* sin.(z)
ψ³ = -sin.(x) .* sin.(z)
ψ⁴ = -cos.(x) .* sin.(z)

∂z = kron(Dz, I + zeros(N, N))
∂x = kron(I + zeros(M + 1, M + 1), Dx)

e1 = norm(∂x * ψ¹[:] - ψ²[:])
println("The error in take in the partial with respect to x is ", e1)
e2 = norm(∂z * (ψ¹[:]) - (sin.(x).*cos.(z))[:])
println("The error in taking the partial with respect to y is ", e2)

zlifted = sparse(kron(Diagonal(z[:]), I + zeros(N, N)))

answ = Dz * z[:]
bDz = copy(Dz)
bDz[1, :] .= 0.0
bDz[1, 1] = 1.0
bDz \ answ - z[:]

zlifted2 = sparse(kron(I + zeros(N, N), Diagonal(z[:])))


# need to test anti-derivative

# boundary indices
b∂z = copy(∂z)
∂Ω¹ = zlifted.rowval[zlifted.nzval.==z[1]]
∂Ω² = zlifted.rowval[zlifted.nzval.==z[end]]
for ∂i in ∂Ω¹
    b∂z[∂i, :] .= 0.0
    b∂z[∂i, ∂i] = 1.0
end

numerical_answ = b∂z \ ones(size(∂z)[2])
exact_answ = (zeros(M, 1).+z)[:]
e1 = maximum(abs.((numerical_answ-exact_answ)[:]))
println("The maximum error is ", e1)

## boundary indices
b∂z = copy(∂z)
∂Ω¹ = zlifted.rowval[zlifted.nzval.==z[1]]
∂Ω² = zlifted.rowval[zlifted.nzval.==z[end]]
for ∂i in ∂Ω²
    b∂z[∂i, :] .= 0.0
    b∂z[∂i, ∂i] = 1.0
end

numerical_answ = b∂z \ ones(size(∂z)[2])
exact_answ = (zeros(M, 1).+z.+2)[:]
e2 = maximum(abs.((numerical_answ-exact_answ)[:]))
println("The maximum error is ", e2)

advection_operator(ψ, ∂x, ∂z) = ∂z * Diagonal(ψ[:]) * ∂x - ∂x * Diagonal(ψ[:]) * ∂z

A¹ = advection_operator(ψ¹, ∂x, ∂z)
A² = advection_operator(ψ², ∂x, ∂z)
A³ = advection_operator(ψ³, ∂x, ∂z)
A⁴ = advection_operator(ψ⁴, ∂x, ∂z)

tmp = sparse(A⁴)
println(length(tmp.nzval))
droprelativezeros!(tmp)
println(length(tmp.nzval))

##
H = Dx * Dx - I
rhs = sin.(x[:]) + sin.(k[2] * x[:])
answ = H \ rhs
exact_answ = -sin.(x[:]) / 2 + sin.(k[2] * x[:]) / (-1 - k[2]^2)
norm(ans - exact_answ)
