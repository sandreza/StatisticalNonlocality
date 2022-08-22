using LinearAlgebra, Random, ProgressBars, GLMakie, Statistics, Random, Distributions
import StatisticalNonlocality: ou_transition_matrix

k = 0.1 # wavenumber
κ = 0.1 # diffusivity
λ = 0.0 # relaxation rate 
γ = 0.1 # ou relaxation: default = 1.0
ϵ = 1.0 # noise strength: default = √2

N = 10 # number of markov states - 1, numerically unstable for large N

# construct markov approximation 
Δx = 2 / √N
uₘ = 1 / sqrt(γ * 2/ϵ^2) * [Δx * (i - N / 2) for i in 0:N]
Q = ou_transition_matrix(N) .* γ
Λ, V = eigen(Q)
V⁻¹ = inv(V)

# define the effective diffusivity as the appropriate schur-complement
U = V * Diagonal(uₘ) * V⁻¹
vtop = U[end, 1:end-1]
vbottom = U[1:end-1, end]
vbot = im * k * U[1:end-1, 1:end-1] + Diagonal(Λ[1:end-1] .- λ .- κ * k^2)
𝒦ₘ = -real(vtop' * (vbot \ vbottom))

##

# Not sure if this is totally sensible
dl = reverse([1.0 * n for n in 1:N])
du = [1.0 for n in 1:N]
d = zeros(N + 1)
Uₕ = 1 / sqrt(γ * 2/ϵ^2) .* Tridiagonal(dl, d, du) # Hermite Polynomial U, position operator in spectral space

ll, vv = eigen(Array(Uₕ)) # nodal positions
QH = vv * Diagonal(Λ) * inv(vv) # nodal space matrix
vtop = Uₕ[end, 1:end-1]
vbottom = Uₕ[1:end-1, end]
vbot = im * k * Uₕ[1:end-1, 1:end-1] + Diagonal(Λ[1:end-1] .- λ .- κ * k^2)
𝒦ₕ = -real(vtop' * (vbot \ vbottom))

##
# now just simulate the OU system numerically, (should do more stable timestepping)
Random.seed!(12345)

Δt = minimum([0.1, 0.1 / γ, 0.5 * (1 / (κ * k^2)), 0.5 * 1 / λ, 0.1 / (k * maximum(uₘ))])

totes = 1:floor(Int, 100000 / Δt)
u = Float64[]
sc = Vector{Float64}[]

push!(u, 0.0)
push!(sc, [0.0, 1.0])
for i in ProgressBar(totes)
    u₊ = u[i] + Δt * (-γ * u[i]) + ϵ * randn() * √Δt
    push!(u, u₊)
    s = sc[i][1]
    c = sc[i][2]
    s₊ = s + Δt * (-(κ * k^2 + λ) * s + c * k * u[i] + 1.0)
    c₊ = c + Δt * (-(κ * k^2 + λ) * c - s * k * u[i])
    push!(sc, [s₊, c₊])
end

ss = [s[1] for s in sc]
cs = [s[2] for s in sc]

fig = Figure()
ax = Axis(fig[1, 1], xlabel="time")
lines!(ax, ss[end-10000:end], color=:red, label="sine mode")
lines!(ax, cs[end-10000:end], color=:blue, label="cosine mode")

# compute the effective diffusivity from the simulation
𝒦 = -mean(u[100:end] .* cs[100:end]) / (k * mean(ss[100:end]))

##
println("Markov estimate: ", 𝒦ₘ)
println("Timeseries estimate: ", 𝒦)
println("Relative Differences: ", abs(𝒦ₘ - 𝒦) / (𝒦ₘ + 𝒦) * 2 * 100, " percent")

##
Random.seed!(10001)
# number_of_states
# simulate a continuous time markov process
T = Q
number_of_states = size(Q)[1]

eT = exp(Δt * T) # get the transition probabilities

ceT = cumsum(eT, dims=1)

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

M = length(totes)
markov_chain = zeros(Int, M)
markov_chain[1] = number_of_states
for i = 2:M
    markov_chain[i] = next_state(markov_chain[i-1], ceT)
end

u = Float64[]
sc = Vector{Float64}[]
push!(u, uₘ[markov_chain[1]])
push!(sc, [0.0, 1.0])
for i in ProgressBar(totes)
    u₊ = uₘ[markov_chain[i]]
    push!(u, u₊)
    s = sc[i][1]
    c = sc[i][2]
    s₊ = s + Δt * (-(κ * k^2 + λ) * s + c * k * u[i] + 1.0)
    c₊ = c + Δt * (-(κ * k^2 + λ) * c - s * k * u[i])
    push!(sc, [s₊, c₊])
end

ss = [s[1] for s in sc]
cs = [s[2] for s in sc]

fig = Figure()
ax = Axis(fig[1, 1], xlabel="time")
lines!(ax, ss[end-1000:end], color=:red, label="sine mode")
lines!(ax, cs[end-1000:end], color=:blue, label="cosine mode")

# compute the effective diffusivity from the simulation
𝒦 = -mean(u[100:end] .* cs[100:end]) / (k * mean(ss[100:end]))

println("Markov estimate: ", 𝒦ₘ)
println("Timeseries estimate: ", 𝒦)
println("Relative Differences: ", abs(𝒦ₘ - 𝒦) / (𝒦ₘ + 𝒦) * 2 * 100, " percent")