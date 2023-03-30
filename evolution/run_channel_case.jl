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
Mensemble = M
c = -1.0
ϵ = 1.0
U = 1.0
tend = 10.0
cfl = minimum([U, 1.0])

## Continuous Case
(; ψ, x, y, kˣ, kʸ, θs, us, vs, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ, θ̇s, θ̅, k₁, k₂, k₃, k₄, θ̃, Δt) = continuous_channel(N, M; c=c, ϵ=ϵ, U = U)
rhs! = n_state_rhs_symmetric_ensemble!
simulation_parameters = (; us, vs, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ)
Δt = cfl * Δt
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
println("The sampling error is ", sampling_error, " percent")
## N-State Case
Ms = collect(4:16)
maxerror = zeros(length(Ms))
l2error = zeros(length(Ms))
nstate_ensemble_means = zeros(N, N, length(Ms))
for (j,M) in ProgressBar(enumerate(Ms))
    (; θs, p, Q, us, vs, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ, θ̇s, θ̅, k₁, k₂, k₃, k₄, θ̃, local_diffusivity_tensor, Δt) = nstate_channel(N, M; c=c, ϵ=ϵ, U = U)
    simulation_parameters = (; p, Q, us, vs, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ)
    Δt = cfl * Δt
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
# scatter(mean(real.(ifft(∂y .* fft(κyx))), dims = 1)[:])
(; θs, p, Q, us, vs, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ, θ̇s, θ̅, k₁, k₂, k₃, k₄, θ̃, local_diffusivity_tensor, Δt) = nstate_channel(N, M; c=c, ϵ=ϵ, U = U)
simulation_parameters = (; κxx, κxy, κyx, κyy, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ)
## Timestep 
Δt_local = cfl * 0.01 * Δt
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
stream_function = @. U * cos(kˣ[2] * x) * cos(kʸ[2] * y)
@info "saving data for 2D channel simulation"
hfile = h5open(pwd() * "/data/channel.hdf5", "w")
hfile["empirical"] = empirical_ensemble_mean
hfile["equations"] = nstate_ensemble_means
hfile["local"] = nstate_ensemble_mean_local
hfile["x"] = x
hfile["y"] = y
hfile["sampling error"] = sampling_error
hfile["local error"] = local_error
hfile["l2 error"] = l2error
hfile["Δt"] = Δt
hfile["iterations"] = iend
hfile["T"] = tend
hfile["Ms"] = Ms
hfile["ensemble size"] = Mensemble
hfile["c"] = c
hfile["epsilon"] = ϵ
hfile["U"] = U
hfile["source"] = real.(s)
hfile["stream function"] = stream_function
close(hfile)
@info "done saving data for channel"
