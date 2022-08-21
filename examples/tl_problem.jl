using LinearAlgebra, Random, ProgressBars, GLMakie, Statistics
import StatisticalNonlocality: ou_transition_matrix

k = 10.0 # wavenumber
κ = 1.0 # diffusivity
λ = 0.0 # relaxation rate

N = 20
Δx = 2/√N
uₘ = [Δx * (i - N/2) for i in 0:N]
Q = ou_transition_matrix(N)
Λ, V = eigen(Q)
V⁻¹ = inv(V)

U = V * Diagonal(uₘ) * V⁻¹
vtop = U[end, 1:end-1]
vbottom = U[1:end-1, end]
vbot = im * k * U[1:end-1, 1:end-1] + Diagonal(Λ[1:end-1] .- λ .- κ * k^2)
𝒦ₘ = -real(vtop' * (vbot \ vbottom))


##

Δt = minimum([0.1, 0.1 * (1/ (κ * k^2))])
γ = 1.0 
ϵ = √2

totes = 1:floor(Int,10000/Δt)
u = Float64[]
sc = Vector{Float64}[]

push!(u, 0.0)
push!(sc, [0.0, 1.0])
for i in ProgressBar(totes)
    u₊ = u[i] + Δt * (- γ * u[i] ) + ϵ * randn() * √Δt
    push!(u, u₊)
    s = sc[i][1]
    c = sc[i][2]
    s₊ = s + Δt * (- (κ * k^2 + λ) * s + c * k *  u[i] + 1.0)
    c₊ = c + Δt * (- (κ * k^2 + λ) * c - s * k *  u[i] )
    push!(sc, [s₊, c₊])
end

ss = [s[1] for s in sc]
cs = [s[2] for s in sc]

fig = Figure()
ax = Axis(fig[1, 1], xlabel="time")
lines!(ax, ss, label="u")

𝒦 =  -mean(u .* cs) / (k * mean(ss))

##
println("Markov estimate: ", 𝒦ₘ)
println("Timeseries estimate: ", 𝒦)
println("Relataive Differences: ", abs(𝒦ₘ - 𝒦) / (𝒦ₘ + 𝒦) * 2 * 100, "%")