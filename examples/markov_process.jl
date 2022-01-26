using StatisticalNonlocality, Distributions, Random, LinearAlgebra
import StatisticalNonlocality: ou_transition_matrix
import StatisticalNonlocality: uniform_phase
import Distributions: Uniform

# Set random seed for reproducibility 
Random.seed!(10001)

# simulate a continuous time markov process
T = ou_transition_matrix(4)
T = uniform_phase(4)

γ = 0.01
eT = exp(γ * T) # get the transition probabilities
# each column gives the transition probability
# column i, means, given that I am in state i, each row j gives the probability to transition to state j

ceT = cumsum(eT, dims = 1)

# Define the jump map
function next_state(current_state_index::Int, cT)
    vcT = view(cT, :, current_state_index)
    u = rand(Uniform(0, 1))
    # choose a random uniform variable and decide next state
    # depending on where one lies on the line with respect to the probability 
    # of being found between given probabilities
    for i in eachindex(vcT)
        if u < vcT[i]
            return i
        end
    end

    return i
end

M = 1000
markov_chain = zeros(Int, M)
markov_chain[1] = 4
for i = 2:M
    markov_chain[i] = next_state(markov_chain[i-1], ceT)
end

using GLMakie
fig2 = scatter(markov_chain)

fig = hist(markov_chain)

# E[u(x⃗, t + τ) * u(x⃗, t) ] - E[u(x⃗,t)]E[u(x⃗,t+τ)] =  ∑ₘ( Pₘ(t) uₘ(x⃗) * ∑ₙ( exptT[n, m] uₙ(x⃗) ) ) - (∑ₘ Pₘ(t) uₘ) (∑ₘ Pₘ(t + τ) uₘ)
# under more assumptions
# ∑ₘₙ ( Pₘ(t) ( exptT[n, m]uₘ(x⃗) uₙ(x⃗) ) ) = u⃗' exp(T) * diag(P) * u⃗

# anti-deriv  T⁻¹ ( I - exp(t T) ) * 

exp(T) * ones(4) ./ 4

exp(3 * T) * [1 -1 1 -1]'
##
basic(α) = [α 1-α; 1-α α]

# {-1, 1}
α = 2 / 3
tmp = basic(α)

# E[X[n]X[n+d]] = (2α - 1)ᵈ
# α = 2/3 => 3⁻ᵈ
sum(tmp * [1, 0] * Diagonal([-1 1])) + sum(tmp * [0, 1] * Diagonal([-1 1]))

# 1/2 * {-1} * (-1 * α + 1 * (1-α)) + 1/2 * {1} * (-1 * (1-α) + 1 * α)
1 / 2 * (-1) * (-1 * α + 1 * (1 - α)) + 1 / 2 * (1) * (-1 * (1 - α) + 1 * α)
[-1 1] * tmp^100 * [-1 / 2, 1 / 2]

[-1 , 1 ]' * tmp * [-1 / 2, 1 / 2]
λ, V = eigen(tmp)
Λ = Diagonal(λ)
V⁻¹ = inv(V)
tmpmat = zeros(size(tmp))
for i = 1:length(λ)
    tmpmat += λ[i] * V[:, i] * transpose(V⁻¹[i, :])
end

## 
V⁻¹ = inv(V)
V * Λ * V⁻¹ - tmp
Diagonal([0, 1])
V * Λ * V⁻¹
V * Diagonal([0, 1]) * V⁻¹
V[:, 2] * V⁻¹[2, :]'

##
λ, V = eigen(T)
Λ = Diagonal(λ)
V⁻¹ = inv(V)
tmp = zeros(length(λ))
tmp[end] = 1
Λ∞ = Diagonal(tmp)
V * Λ * V⁻¹ - T
real.(V * Λ∞ * V⁻¹)
real.(V[:, end] * transpose(V⁻¹[end, :]))

tmpmat = zeros(ComplexF64, size(T))

for i = 1:length(λ)
    tmpmat += λ[i] * V[:, i] * transpose(V⁻¹[i, :])
end

T̃ = real.(tmpmat)
norm(T̃ - T)
