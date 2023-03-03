using LinearAlgebra

# Base transition matrix
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

# The following transformation is what is used
S = real.(V)
S[:, 1] .= real.(W¹)
S[:, 2] .= real.(0.5 * (W² + W³))
S[:, 3] .= real.(0.5 * im * (W² - W³))
S[:, 4] .= real.(W⁴)

display(S)
display(inv(S))
display(inv(S) * T * S)

# The following partitions the magnitude of
# the anti-symmetric component and the symmetric component
# of the transition matrix
sT = (T + T') / 2
aT = (T - T') / 2
λ = 1.0
ω = 1.0
inv(S) * (λ * sT + ω * aT) * S

## check kronecker products corresponding to random phase
kroneckered_phase = kron(sT, I + 0 * sT) + kron(I + 0 * sT, sT)
Λ, V = eigen(kroneckered_phase)

## check out higher dimensional one
idmat = I + 0 * sT # the identity matrix
kroneckered_phase = kron(sT, idmat, idmat) + kron(idmat, sT, idmat) + kron(idmat, idmat, sT)
Λ, V = eigen(kroneckered_phase)

##
import StatisticalNonlocality: ou_transition_matrix

T = ou_transition_matrix(10)
P = exp(0.1 .* T)
Λ, V = eigen(P)
p = real.(V[:, end] ./ sum(V[:, end]))
entropy = 0.0

for i in eachindex(p), j in eachindex(p)
    global entropy += -p[j] * P[i, j] * log(P[i, j]) / log(2)
end
entropy

entropy2 = 0.0
P² = P * P
for i in eachindex(p), j in eachindex(p)
    global entropy2 += -p[j] * P²[i, j] * log(P²[i, j]) / log(2)
end
entropy2

entropy∞ = 0.0
P∞ = p * ones(length(p))'
for i in eachindex(p), j in eachindex(p)
    global entropy∞ += -p[j] * P∞[i, j] * log(P∞[i, j]) / log(2)
end
entropy∞ 

println("The entropies are ")
println(entropy)
println(entropy2)
println(entropy∞)