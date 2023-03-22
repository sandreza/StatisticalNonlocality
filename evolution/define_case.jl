using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2
using StatisticalNonlocality: ou_transition_matrix, uniform_phase, advection_matrix_central, discrete_laplacian_periodic
using MarkovChainHammer.TransitionMatrix: steady_state
using StatisticalNonlocality, Distributions
using ProgressBars
rng = MersenneTwister(1234)
Random.seed!(123456789)
arraytype = Array
include("conditional_mean_evolution.jl")
include("allocate_fields.jl")
include("nstate_cases.jl")
include("continuous_cases.jl")

N = 2^5 # number of gridpoints
M = 10000   # number of states
c = -π/2
ϵ = π/sqrt(8)
U = 1.0
tend = 10.0

## Continuous Case
(; ψ, x, y, kˣ, kʸ, θs, us, vs, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ, θ̇s, θ̅, k₁, k₂, k₃, k₄, θ̃, Δt) = continuous_channel(N, M; c=c, ϵ=ϵ)
rhs! = n_state_rhs_symmetric_ensemble!
simulation_parameters = (; us, vs, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ)
iend = ceil(Int, tend / Δt)
process = periodic_drift(c, ϵ, Δt, M, iend)
# runge kutta 4 timestepping
for i in ProgressBar(1:iend)
    update_channel_flow_field!(us, vs, ψ, process[:, i], x, y, kˣ, kʸ, ∂y, ∂x, P, P⁻¹, U)
    rhs!(k₁, θs, simulation_parameters)
    [θ̃[i] .= θs[i] .+ Δt * k₁[i] * 0.5 for i in eachindex(θs)]
    rhs!(k₂, θ̃, simulation_parameters)
    [θ̃[i] .= θs[i] .+ Δt * k₂[i] * 0.5 for i in eachindex(θs)]
    rhs!(k₃, θ̃, simulation_parameters)
    [θ̃[i] .= θs[i] .+ Δt * k₃[i] * 0.5 for i in eachindex(θs)]
    rhs!(k₄, θ̃, simulation_parameters)
    [θs[i] .+= Δt / 6 * (k₁[i] + 2 * k₂[i] + 2 * k₃[i] + k₄[i]) for i in eachindex(θs)]
end
empirical_ensemble_mean = real.(mean(θs))
empirical_ensemble_standard_deviation = real.(std(θs))
sampling_error = norm(mean(θs[1:floor(Int, M / 2)]) - mean(θs[floor(Int, M / 2)+1:end])) / norm(empirical_ensemble_mean)
println("maximum value of theta after ", maximum(empirical_ensemble_mean))
println("The sampling error is ", sampling_error)
## N-State Case
Ms = collect(4:16)
maxerror = zeros(length(Ms))
l2error = zeros(length(Ms))
nstate_ensemble_means = zeros(N, N, length(Ms))
for (j,M) in ProgressBar(enumerate(Ms))
    (; θs, p, Q, us, vs, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ, θ̇s, θ̅, k₁, k₂, k₃, k₄, θ̃, local_diffusivity_tensor, Δt) = nstate_channel(N, M; c=c, ϵ=ϵ)
    simulation_parameters = (; p, Q, us, vs, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ)
    ## Timestep 
    rhs! = n_state_rhs_symmetric!
    iend = ceil(Int, tend / Δt)
    # runge kutta 4 timestepping
    for i in ProgressBar(1:iend)
        rhs!(k₁, θs, simulation_parameters)
        [θ̃[i] .= θs[i] .+ Δt * k₁[i] * 0.5 for i in eachindex(θs)]
        rhs!(k₂, θ̃, simulation_parameters)
        [θ̃[i] .= θs[i] .+ Δt * k₂[i] * 0.5 for i in eachindex(θs)]
        rhs!(k₃, θ̃, simulation_parameters)
        [θ̃[i] .= θs[i] .+ Δt * k₃[i] * 0.5 for i in eachindex(θs)]
        rhs!(k₄, θ̃, simulation_parameters)
        [θs[i] .+= Δt / 6 * (k₁[i] + 2 * k₂[i] + 2 * k₃[i] + k₄[i]) for i in eachindex(θs)]
    end
    nstate_ensemble_mean = real.(sum(θs))
    nstate_ensemble_means[:,:, j] .= nstate_ensemble_mean
    println("maximum value of theta after ", maximum(nstate_ensemble_mean))
    maxerror[j] = maximum(abs.(empirical_ensemble_mean .- nstate_ensemble_mean)) / maximum(empirical_ensemble_mean) * 100
    l2error[j] = norm(empirical_ensemble_mean .- nstate_ensemble_mean) / norm(empirical_ensemble_mean) * 100
    # println("maximum error = ", maxerror[j], " percent relative error")
    # println("l2 error = ", l2error[j], " percent relative error")
end
##
using GLMakie
error_fig = Figure()
ax11 = Axis(error_fig[1, 1]; xlabel="M", ylabel="Maximum Error (%)", xticklabelsize=20, yticklabelsize=20)
scatter!(ax11, Ms, maxerror)
ylims!(ax11, (0, 50))
ax12 = Axis(error_fig[1, 2]; xlabel="M", ylabel="L2 Error (%)", xticklabelsize=20, yticklabelsize=20)
scatter!(ax12, Ms, l2error)
ylims!(ax12, (0, 50))
ax13 = Axis(error_fig[1, 3]; xlabel="M", ylabel="L2 Self Error (%)", xticklabelsize=20, yticklabelsize=20)
scatter!(ax13, Ms, [norm(nstate_ensemble_means[:, :, i] .- nstate_ensemble_means[:, :, end]) for i in 1:length(Ms)])
ylims!(ax13, (0, 50))
display(error_fig)
##
Nd2 = floor(Int, N / 2) + 1
fig = Figure(resolution=(2100, 1000))
titlelables1 = ["Simulated"]
options = (; titlesize=30, xlabel="x", ylabel="y", xlabelsize=40, ylabelsize=40, xticklabelsize=30, yticklabelsize=30)
colorrange = (-0.15, 0.15)
ax = Axis(fig[1, 1]; title=titlelables1[1], options...)
index_choice = length(Ms)
field_cont = nstate_ensemble_means[:, 1:Nd2, index_choice]
heatmap!(ax, x[:], y[1:Nd2], field_cont, colormap=:balance, interpolate=true, colorrange = colorrange)
contour!(ax, x[:], y[1:Nd2], field_cont, color=:black, levels=10, linewidth=1.0)
ax = Axis(fig[2, 1]; title="Empirical", options...)
field_tmp = empirical_ensemble_mean[:, 1:Nd2]
heatmap!(ax, x[:], y[1:Nd2], field_tmp, colormap=:balance, interpolate=true, colorrange = colorrange)
contour!(ax, x[:], y[1:Nd2], field_tmp, color=:black, levels=10, linewidth=1.0)
ax = Axis(fig[3, 1]; title="Difference", options...)
field_tmp = empirical_ensemble_mean[:, 1:Nd2] - nstate_ensemble_means[:, 1:Nd2, index_choice]
heatmap!(ax, x[:], y[1:Nd2], field_tmp, colormap=:balance, interpolate=true, colorrange = colorrange)
contour!(ax, x[:], y[1:Nd2], field_tmp, color=:black, levels=10, linewidth=1.0)
display(fig)