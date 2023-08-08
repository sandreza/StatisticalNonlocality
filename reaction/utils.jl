using StatisticalNonlocality, LinearAlgebra, Statistics
using MarkovChainHammer, FFTW, GLMakie, ProgressBars
using MarkovChainHammer.TransitionMatrix: steady_state
import StatisticalNonlocality: ou_transition_matrix
include("allocate_fields.jl")

# Construct Markov Model
N¹ = 3 # number of markov states
γ = 1.0 # ou relaxation: default = 1.0
ϵ = sqrt(2) # noise strength: default = √2
N = N¹-1 # number of markov states - 1, numerically unstable for large N
# construct markov approximation 
if N == 0
    κ = 1e-0 # 1e-3
    us = [0.0]
    p = [1.0]
    Q = [0.0]
else
    κ = 0 # 1e-3 # 1e-3
    Δx = 2 / √N
    us = 1 / sqrt(γ * 2 / ϵ^2) * [Δx * (i - N / 2) for i in 0:N]
    Q = ou_transition_matrix(N) .* γ
    Λ, V = eigen(Q)
    p = steady_state(Q) # from markov chain hammer
end
# Allocate Fields
M = 32
field_tuples = allocate_fields(M, N¹; arraytype = Array)
# Construct Domain 
Ω = S¹(2π) 
grid = FourierGrid(M, Ω)
nodes, wavenumbers = grid.nodes, grid.wavenumbers
x = nodes[1]
k = wavenumbers[1]
∂x = im * k
Δ = @. ∂x^2
P = plan_fft!(field_tuples.θs[1])
P⁻¹ = plan_ifft!(field_tuples.θs[1])
# Define Simulation Parameters
λ = 0.01 # 0.01
δ = 0.7

# Construction simulation parameters
simulation_parameters = (; p, Q, ∂x, Δ, us, P, P⁻¹, κ, λ, field_tuples...)
(; θ̇s, θs, c⁰) = field_tuples #extract
@. c⁰ = 1 - δ * cos(k[2] * x)
# Initialize with c⁰ 
[θ .= c⁰ * p[i] for (i,θ) in enumerate(θs)]
##
cauchy_criteria = 1e-7
dt = minimum([0.25/(M  * sqrt(N¹)), 1/(M^2 * κ)]) 
mean_theta = Float64[]
for i in ProgressBar(1:1000000)
    rhs!(θ̇s, θs, simulation_parameters)
    @. θs += θ̇s * dt
    if any(isnan.(θs[1]))
        println("nan")
        break
    end
    push!(mean_theta, mean(real.(sum(θs))))
    if i > 100000 
        if abs(mean_theta[i] - mean_theta[i-100])/(mean_theta[i]) < cauchy_criteria
            println("converged")
            break
        end
    end
end
##
fig = Figure()
ax = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])
lines!(ax, real.(sum(θs)), color = :red)
lines!(ax, real.(c⁰), color = :blue)
for i in 1:N¹
    lines!(ax2, real.(θs[i]))
end
display(fig)
# lines(real.(θs[1]))
# plot(mean_theta)
##
θ̄ = real.(sum(θs))
println(mean(θ̄))
println(extrema(θ̄))
sqrt(1 - δ^2)