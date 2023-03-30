using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2, HDF5
using StatisticalNonlocality: ou_transition_matrix, uniform_phase, advection_matrix_central, discrete_laplacian_periodic
using MarkovChainHammer.TransitionMatrix: steady_state
using MarkovChainHammer.Trajectory: generate
using StatisticalNonlocality, Distributions
using ProgressBars
rng = MersenneTwister(1234)
Random.seed!(123456789)
arraytype = Array
include("conditional_mean_evolution.jl")
include("allocate_fields.jl")
include("nstate_cases.jl")
include("continuous_cases.jl")

N = 48 # number of gridpoints
M = 10000 # number of ensembles 
Mens = M
nstate = 3 # number of states
γ = 1
ϵ = √2
U = 1.0
tend = 25.0
cfl = minimum([U, 1.0])

## N-State Case
(; ψ, uₘ, kˣ, kʸ,x, y, θs, p, Q, us, vs, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ, θ̇s, θ̅, k₁, k₂, k₃, k₄, θ̃, local_diffusivity_tensor, Δt) = nstate_ou(N, M; γ=γ, ϵ=ϵ, arraytype=arraytype, nstate = nstate)
simulation_parameters = (; p,Q, us, vs, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ)
# -uₘ' * pinv(Q) * Diagonal(steady_state(Q)) * uₘ # autocorrelation
rhs! = n_state_rhs_symmetric_ensemble! # n_state_rhs_symmetric!
Δt = cfl * Δt
iend = ceil(Int, tend / Δt)
process, process_index = nstate_ou_process(Q, uₘ, Δt, M, iend)
# runge kutta 4 timestepping
for i in ProgressBar(1:iend)
    update_ou_flow_field!(us, vs, ψ, process[:, i], x, y, kˣ, kʸ, ∂y, ∂x, P, P⁻¹) 
    rhs!(k₁, θs, simulation_parameters)
    [θ̃[i] .= θs[i] .+ Δt * k₁[i] * 0.5 for i in eachindex(θs)]
    rhs!(k₂, θ̃, simulation_parameters)
    [θ̃[i] .= θs[i] .+ Δt * k₂[i] * 0.5 for i in eachindex(θs)]
    rhs!(k₃, θ̃, simulation_parameters)
    [θ̃[i] .= θs[i] .+ Δt * k₃[i] * 0.5 for i in eachindex(θs)]
    rhs!(k₄, θ̃, simulation_parameters)
    [θs[i] .+= Δt / 6 * (k₁[i] + 2 * k₂[i] + 2 * k₃[i] + k₄[i]) for i in eachindex(θs)]
end
Θⁱ = zeros(N, N, nstate)
for i in 1:M
    Θⁱ[:, :, process_index[i, end]] .+= real.(θs[i] / M)
end

empirical_ensemble_mean = real.(mean(θs))
empirical_ensemble_standard_deviation = real.(std(θs))
empirical_θs = copy(θs)
sampling_error = norm(mean(empirical_θs[1:floor(Int, M / 2)]) - mean(empirical_θs[floor(Int, M / 2)+1:end])) / norm(empirical_ensemble_mean) * 100

println("maximum value of theta after ", maximum(empirical_ensemble_mean))
println("The sampling error is ", sampling_error, " percent")
## N-State Case
M = 3 # number of members
nstate = 3 # number of states
## N-State Case
(; ψ, uₘ, kˣ, kʸ, x, y, θs, p, Q, us, vs, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ, θ̇s, θ̅, k₁, k₂, k₃, k₄, θ̃, local_diffusivity_tensor, Δt) = nstate_ou(N, M; γ=γ, ϵ=ϵ, arraytype=arraytype, nstate=nstate)
simulation_parameters = (; p, Q, us, vs, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ)
# -uₘ' * pinv(Q) * Diagonal(steady_state(Q)) * uₘ # autocorrelation
rhs! = n_state_rhs_symmetric!
Δt = cfl * Δt
iend = ceil(Int, tend / Δt)
process, process_index = nstate_ou_process(Q, uₘ, Δt, M, iend)
# runge kutta 4 timestepping
for i in ProgressBar(1:iend)
    # update_ou_flow_field!(us, vs, ψ, process[:, i], x, y, kˣ, kʸ, ∂y, ∂x, P, P⁻¹)
    rhs!(k₁, θs, simulation_parameters)
    [θ̃[i] .= θs[i] .+ Δt * k₁[i] * 0.5 for i in eachindex(θs)]
    rhs!(k₂, θ̃, simulation_parameters)
    [θ̃[i] .= θs[i] .+ Δt * k₂[i] * 0.5 for i in eachindex(θs)]
    rhs!(k₃, θ̃, simulation_parameters)
    [θ̃[i] .= θs[i] .+ Δt * k₃[i] * 0.5 for i in eachindex(θs)]
    rhs!(k₄, θ̃, simulation_parameters)
    [θs[i] .+= Δt / 6 * (k₁[i] + 2 * k₂[i] + 2 * k₃[i] + k₄[i]) for i in eachindex(θs)]
end
Θₘ = zeros(N, N, nstate)
for i in 1:M
    Θₘ[:, :, i] .= real.(θs[i])
end

markov_ensemble_mean = real.(sum(θs))
println("maximum value of theta after ", maximum(markov_ensemble_mean))
##
stream_function = @. cos(kˣ[2] * x) * cos(kʸ[2] * y)
@info "saving data for 2D, 3-State OU"
hfile = h5open(pwd() * "/data/3_state_ou.hdf5", "w")
hfile["empirical"] = Θⁱ
hfile["equations"] = Θₘ
hfile["x"] = x
hfile["y"] = y
hfile["sampling error"] = sampling_error
hfile["Δt"] = Δt
hfile["iterations"] = iend
hfile["T"] = tend
hfile["nstates"] = nstate
hfile["ensemble size"] = Mens
hfile["gamma"] = γ
hfile["epsilon"] = ϵ
hfile["uₘ"] = uₘ
hfile["source"] = real.(s)
hfile["stream function"] = stream_function
close(hfile)
@info "done saving data for ou"