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
M = 10000  # number of states
c = -1.0
ϵ = 1.0
U = 2.0
tend = 10.0

## Continuous Case
(; ψ, x, y, kˣ, kʸ, θs, us, vs, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ, θ̇s, θ̅, k₁, k₂, k₃, k₄, θ̃, Δt) = continuous_channel(N, M; c=c, ϵ=ϵ, U = U)
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
empirical_θs = copy(θs)
sampling_error = norm(mean(empirical_θs[1:floor(Int, M / 2)]) - mean(empirical_θs[floor(Int, M / 2)+1:end])) / norm(empirical_ensemble_mean) * 100

println("maximum value of theta after ", maximum(empirical_ensemble_mean))
println("The sampling error is ", sampling_error)
## N-State Case
Ms = collect(4:16)
maxerror = zeros(length(Ms))
l2error = zeros(length(Ms))
nstate_ensemble_means = zeros(N, N, length(Ms))
for (j,M) in ProgressBar(enumerate(Ms))
    (; θs, p, Q, us, vs, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ, θ̇s, θ̅, k₁, k₂, k₃, k₄, θ̃, local_diffusivity_tensor, Δt) = nstate_channel(N, M; c=c, ϵ=ϵ, U = U)
    simulation_parameters = (; p, Q, us, vs, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ)
    Δt = Δt
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
## Local Diffusivity Case 
N = 2^5 # number of gridpoints
M = 1   # number of states
Nstates = 100 # for local diffusivity estimate
(; θs, p, Q, us, vs, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ, θ̇s, θ̅, k₁, k₂, k₃, k₄, θ̃, local_diffusivity_tensor, Δt) = nstate_channel(N, Nstates; c=c, ϵ=ϵ, U = U)
κxx = 0.0 .+ 1 * local_diffusivity_tensor[:, :, 1, 1]
κyy = 0.0 .+ 1 * local_diffusivity_tensor[:, :, 2, 2] 
κyx = 0.0 .+ 1 * local_diffusivity_tensor[:, :, 2, 1] 
κxy = 0.0 .+ 1 * local_diffusivity_tensor[:, :, 1, 2]
(; θs, p, Q, us, vs, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ, θ̇s, θ̅, k₁, k₂, k₃, k₄, θ̃, local_diffusivity_tensor, Δt) = nstate_channel(N, M; c=c, ϵ=ϵ, U = U)
simulation_parameters = (; κxx, κxy, κyx, κyy, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ)
## Timestep 
Δt_local = 0.01 * Δt
rhs! = n_state_rhs_symmetric!
iend = ceil(Int, tend / Δt_local)
rhs! = n_state_rhs_local!
# runge kutta 4 timestepping
for i in ProgressBar(1:iend)
    rhs!(k₁, θs, simulation_parameters)
    [θ̃[i] .= θs[i] .+ Δt_local * k₁[i] * 0.5 for i in eachindex(θs)]
    rhs!(k₂, θ̃, simulation_parameters)
    [θ̃[i] .= θs[i] .+ Δt_local * k₂[i] * 0.5 for i in eachindex(θs)]
    rhs!(k₃, θ̃, simulation_parameters)
    [θ̃[i] .= θs[i] .+ Δt_local * k₃[i] * 0.5 for i in eachindex(θs)]
    rhs!(k₄, θ̃, simulation_parameters)
    [θs[i] .+= Δt_local / 6 * (k₁[i] + 2 * k₂[i] + 2 * k₃[i] + k₄[i]) for i in eachindex(θs)]
end
nstate_ensemble_mean_local = real.(sum(θs))
local_error = norm(empirical_ensemble_mean .- nstate_ensemble_mean_local) / norm(empirical_ensemble_mean) * 100
println("maximum value of theta after ", maximum(nstate_ensemble_mean_local))
##
using GLMakie
error_fig = Figure(resolution=(1612, 1180))
options = (; titlesize=30, xlabelsize=40, ylabelsize=40, xticklabelsize=40, yticklabelsize=40)
ax11 = Axis(error_fig[1, 1]; title = "Relative L2 Error", xlabel="Number of States", ylabel="L2 Error (%)", options...)
scatter!(ax11, Ms, l2error; markersize=30, color=:black, label="N-State Model")
hlines!(ax11, [sampling_error], color=:red, linewidth=10, linestyle = :dash, label = "Sampling Error")
hlines!(ax11, [local_error], color=:orange, linewidth=10, linestyle=:dash, label="Local Diffusivity Error")
ylims!(ax11, (0, 80))
ax11.xticks = (collect(Ms), string.(collect(Ms)))
axislegend(ax11, position=:rt, framecolor=(:grey, 0.5), patchsize=(40, 40), markersize=40, labelsize=50)
display(error_fig)
##
Nd2 = floor(Int, N / 2) + 1
fig = Figure(resolution=(2100, 1000))
titlelables1 = ["N = $(Ms[end]) State"]
options = (; titlesize=30, xlabel="x", ylabel="y", xlabelsize=40, ylabelsize=40, xticklabelsize=30, yticklabelsize=30)
mth = maximum(nstate_ensemble_means)
colorrange = (-mth, mth)
ax = Axis(fig[1, 1]; title=titlelables1[1], options...)
index_choice = length(Ms)
field_cont = nstate_ensemble_means[:, 1:Nd2, index_choice]
heatmap!(ax, x[:], y[1:Nd2], field_cont, colormap=:balance, interpolate=true, colorrange = colorrange)
contour!(ax, x[:], y[1:Nd2], field_cont, color=:black, levels=10, linewidth=1.0)
ax = Axis(fig[1, 2]; title="Empirical with 10000 Ensemble Members", options...)
field_tmp = empirical_ensemble_mean[:, 1:Nd2]
heatmap!(ax, x[:], y[1:Nd2], field_tmp, colormap=:balance, interpolate=true, colorrange = colorrange)
contour!(ax, x[:], y[1:Nd2], field_tmp, color=:black, levels=10, linewidth=1.0)
ax = Axis(fig[2, 2]; title="Local Diffusivity", options...)
field_tmp = nstate_ensemble_mean_local[:, 1:Nd2]
heatmap!(ax, x[:], y[1:Nd2], field_tmp, colormap=:balance, interpolate=true, colorrange=colorrange)
contour!(ax, x[:], y[1:Nd2], field_tmp, color=:black, levels=10, linewidth=1.0)
ax = Axis(fig[2, 1]; title="NState - Empirical", options...)
field_tmp = empirical_ensemble_mean[:, 1:Nd2] - nstate_ensemble_means[:, 1:Nd2, index_choice]
heatmap!(ax, x[:], y[1:Nd2], field_tmp, colormap=:balance, interpolate=true, colorrange = colorrange)
contour!(ax, x[:], y[1:Nd2], field_tmp, color=:black, levels=10, linewidth=1.0)
Colorbar(fig[1:2, 3]; limits=colorrange, colormap=:balance, flipaxis=false, ticklabelsize = 30)
display(fig)