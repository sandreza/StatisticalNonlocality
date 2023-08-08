using StatisticalNonlocality, LinearAlgebra, Statistics
using MarkovChainHammer, FFTW, GLMakie, ProgressBars
using MarkovChainHammer.TransitionMatrix: steady_state
import StatisticalNonlocality: ou_transition_matrix
include("allocate_fields.jl")

# Construct Markov Model
N¹ = 2 # number of markov states
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
    κ = 1e-3 # 1e-3 # 1e-3
    wavenumber = 2 # 2 is default
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
λ = 0.5 # 0.01
λs = [10^i for i in -2:0.1:2]
meanlist = Float64[]
oldmeanlist = Float64[]
δ = 0.1
for λ in ProgressBar(λs)

    # Construction simulation parameters
    simulation_parameters = (; p, Q, ∂x, Δ, us, P, P⁻¹, κ, λ, field_tuples...)
    (; θ̇s, θs, c⁰) = field_tuples #extract
    @. c⁰ = 1 - δ * cos(k[wavenumber] * x)
    # Initialize with c⁰ 
    [θ .= c⁰ * p[i] for (i,θ) in enumerate(θs)]
    ##
    cauchy_criteria = 1e-7
    dt = minimum([0.25/(M  * sqrt(N¹)), 1/(M^2 * κ)]) /2
    mean_theta = Float64[]
    for i in ProgressBar(1:1000000)
        rhs2!(θ̇s, θs, simulation_parameters)
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
    φs = deepcopy(θs)
    ##
    [θ .= c⁰ * p[i] for (i,θ) in enumerate(θs)]
    ##
    cauchy_criteria = 1e-7
    dt = minimum([0.25/(M  * sqrt(N¹)), 1/(M^2 * κ)]) /2
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
    ths = deepcopy(θs)
    scatter(real.(sum(θs) .* sum(φs)))
    ##
    mean(sum(θs))
    ##
    (std.([φs[i] .* θs[i] / (p[i]^2) for i in eachindex(θs)]) ./ mean(sum(θs)) ) .* 100
    ##
    simulation_parameters = (; simulation_parameters..., φs)
    [θ .= c⁰ * p[i] for (i,θ) in enumerate(θs)]
    ##
    cauchy_criteria = 1e-7
    dt = minimum([0.25/(M  * sqrt(N¹)), 1/(M^2 * κ)]) /2
    mean_theta = Float64[]
    for i in ProgressBar(1:1000000)
        rhs3!(θ̇s, θs, simulation_parameters)
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

    # scatter(real.(sum(θs)))
    ##
    push!(meanlist, mean(sum(θs)))
    push!(oldmeanlist, mean(sum(ths)))
end

##
#=
fig = Figure()
ax = Axis(fig[1,1])
lines!(ax, real.(sum(θs)))
lines!(ax, real.(sum(ths)))
display(fig)
=#
##
fig = Figure()
fontsize = 40
options = (; xlabelsize = fontsize, ylabelsize = fontsize, xticklabelsize = fontsize, yticklabelsize = fontsize)
ax = Axis(fig[1,1]; options..., xlabel = "log10(λ)", ylabel = "mean")
lines!(ax, log10.(λs), meanlist, linewidth = 10, color = (:blue, 0.5), label = "new")
lines!(ax, log10.(λs), oldmeanlist, linewidth = 10, color = (:red, 0.5), label = "old")
ylims!(ax, (sqrt(1 - δ^2) .* 0.95, 1.05))
axislegend(ax, position = :rb, labelsize = fontsize)
display(fig)