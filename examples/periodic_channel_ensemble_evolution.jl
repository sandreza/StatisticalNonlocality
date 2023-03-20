using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2
using StatisticalNonlocality: ou_transition_matrix, uniform_phase
using MarkovChainHammer.TransitionMatrix: steady_state
using MarkovChainHammer.Trajectory: generate
using StatisticalNonlocality
using ProgressBars
rng = MersenneTwister(1234)
Random.seed!(123456789)

using CUDA
arraytype = Array
Ω = S¹(2π)×S¹(2)
N = 2^5 # number of gridpoints
M = 100 # number of states
U = 1.0 # amplitude factor
T = uniform_phase(4)
γ = 1
ω = 1
Q =  γ * (T + T')/2 + ω * (T - T')/2
p = steady_state(Q)
Λ, V = eigen(Q)
V[:, end] .= p
for i in 1:4-1
    V[:, i] .= V[:, i] ./ norm(V[:, i], 1)
end
V⁻¹ = inv(V)

grid = FourierGrid(N, Ω, arraytype=arraytype)
nodes, wavenumbers = grid.nodes, grid.wavenumbers



x = nodes[1]
y = nodes[2]
kˣ = wavenumbers[1]
kʸ = wavenumbers[2]

##
# Fields 
# velocity
ψ = arraytype(zeros(ComplexF64, N, N))
ψ2 = arraytype(zeros(ComplexF64, N, N))

# theta
θs = [similar(ψ) for i in 1:M]
θv = similar(ψ)
∂ˣθ = similar(ψ)
∂ʸθ = similar(ψ)
κΔθ = similar(ψ)
u = similar(ψ)
v = similar(ψ)
u2 = similar(ψ)
v2 = similar(ψ)
θ̇s = [similar(ψ) .* 0 for i in 1:M]
s = similar(ψ)
θ̅ = similar(ψ)
k₁ = [similar(ψ) for i in 1:M]
k₂ = [similar(ψ) for i in 1:M]
k₃ = [similar(ψ) for i in 1:M]
k₄ = [similar(ψ) for i in 1:M]
θ̃ = [similar(ψ) for i in 1:M]
uθ = similar(ψ)
vθ = similar(ψ)
∂ˣuθ = similar(ψ)
∂ʸvθ = similar(ψ)

# source
s = similar(ψ)
index = 3
@. s = sin(kˣ[index] * x) * sin(kʸ[index] * y) # could also set source term to zero

# operators
∂x = im * kˣ
∂y = im * kʸ
Δ = @. ∂x^2 + ∂y^2
κ = 0.01

# set equal to diffusive solution 
tmp = (kˣ[index]^2 + kʸ[index]^2)
for (i, θ) in enumerate(θs)
    θ .= (s ./ (tmp * κ)) 
end

println("maximum value of theta before ", maximum(real.(sum(θs))))


# plan ffts
P = plan_fft!(ψ)
P⁻¹ = plan_ifft!(ψ)

# set stream function and hence velocity
@. ψ = U * cos(kˣ[2] * x) * sin(kʸ[2] * y)
@. ψ2 = U * sin(kˣ[2] * x) * sin(kʸ[2] * y)
P * ψ  # in place fft
P * ψ2 # in place fft
# ∇ᵖψ
@. u = -1.0 * (∂y * ψ);
@. v = (∂x * ψ);
@. u2 = -1.0 * (∂y * ψ2);
@. v2 = (∂x * ψ2);
P⁻¹ * ψ;
P⁻¹ * ψ2;
P⁻¹ * u;
P⁻¹ * v; # don't need ψ anymore
P⁻¹ * u2;
P⁻¹ * v2; # don't need ψ2 anymore

us = [copy(u) for i in 1:M]
vs = [copy(v) for i in 1:M]
us_base = [u, u2, -u, -u2]
vs_base = [v, v2, -v, -v2]
## timestepping
Δx = x[2] - x[1]
cfl = 1.0 #0.1
advective_Δt = cfl * Δx / maximum(real.(u))
diffusive_Δt = cfl * Δx^2 / κ
transition_Δt = cfl / maximum(-real.(eigvals(Q)))
Δt = min(advective_Δt, diffusive_Δt, transition_Δt)

simulation_parameters = (; p, Q, us, vs, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ)

function n_state_rhs_symmetric_ensemble!(θ̇s, θs, simulation_parameters)
    (; us, vs, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ) = simulation_parameters

    # need A (amplitude), p (probability of being in state), Q (transition probability)
    for (i, θ) in enumerate(θs)
        u = us[i]
        v = vs[i]
        θ̇ = θ̇s[i]
        # dynamics
        P * θ # in place fft
        # ∇θ
        @. ∂ˣθ = ∂x * θ
        @. ∂ʸθ = ∂y * θ
        # κΔθ
        @. κΔθ = κ * Δ * θ
        # go back to real space 
        [P⁻¹ * field for field in (θ, ∂ˣθ, ∂ʸθ, κΔθ)] # in place ifft
        # compute u * θ and v * θ take derivative and come back
        @. uθ = u * θ
        @. vθ = v * θ
        P * uθ
        P * vθ
        @. ∂ˣuθ = ∂x * uθ
        @. ∂ʸvθ = ∂y * vθ
        P⁻¹ * ∂ˣuθ
        P⁻¹ * ∂ʸvθ
        # compute θ̇ in real space
        @. θ̇ = -(u * ∂ˣθ + v * ∂ʸθ + ∂ˣuθ + ∂ʸvθ) * 0.5 + κΔθ + s 
    end

    return nothing
end

n_state_rhs_symmetric_ensemble!(θ̇s, θs, simulation_parameters)
rhs! = n_state_rhs_symmetric_ensemble!
##
PF = exp(Q * Δt)
tend = 10.0
iend = ceil(Int, tend / Δt)
process = [generate(PF, iend) for i in 1:M]

# runge kutta 4 timestepping
for i in ProgressBar(1:iend)
    [us[j] .= us_base[process[j][i]] for j in 1:M]
    [vs[j] .= vs_base[process[j][i]] for j in 1:M]
    rhs!(k₁, θs, simulation_parameters)
    [θ̃[i] .= θs[i] .+ Δt * k₁[i] * 0.5 for i in eachindex(θs)]
    rhs!(k₂, θ̃, simulation_parameters)
    [θ̃[i] .= θs[i] .+ Δt * k₂[i] * 0.5 for i in eachindex(θs)]
    rhs!(k₃, θ̃, simulation_parameters)
    [θ̃[i] .= θs[i] .+ Δt * k₃[i] * 0.5 for i in eachindex(θs)]
    rhs!(k₄, θ̃, simulation_parameters)
    [θs[i] .+= Δt / 6 * (k₁[i] + 2 * k₂[i] + 2 * k₃[i] + k₄[i]) for i in eachindex(θs)]
end

println("maximum value of theta after ", maximum(real.(mean(θs))))
##
using GLMakie
Nd2 = floor(Int, N / 2) + 1
fig = Figure(resolution = (2100, 1000))
titlelables1 = ["Simulated"]
options = (; titlesize=30, xlabel="x", ylabel="y", xlabelsize=40, ylabelsize=40, xticklabelsize=30, yticklabelsize=30)
ax = Axis(fig[1, 1]; title=titlelables1[1], options...)
field = real.(mean(θs))[:, 1:Nd2]
heatmap!(ax, x[:], y[1:Nd2], field, colormap=:balance, interpolate=true)
contour!(ax, x[:], y[1:Nd2], field, color=:black, levels=10, linewidth=1.0)
display(fig)

##
cm = zeros(N, N, 4)
for i in 1:M 
    cm[:, :, process[i][end]] .+= θs[i] / M
end

using GLMakie
Nd2 = floor(Int, N / 2) + 1
fig = Figure(resolution=(2100, 1000))
titlelables1 = ["Θ₁", "Θ₂", "Θ₃", "Θ₄"]
options = (; titlesize=30, xlabel="x", ylabel="y", xlabelsize=40, ylabelsize=40, xticklabelsize=30, yticklabelsize=30)
for i in 1:4
    ax = Axis(fig[1, i]; title=titlelables1[i], options...)
    heatmap!(ax, x[:], y[1:Nd2], cm[:, 1:Nd2, i], colormap=:balance, interpolate=true)
    contour!(ax, x[:], y[1:Nd2], cm[:, 1:Nd2, i], color=:black, levels=10, linewidth=1.0)
end
Random.seed!(12345)
process_index =  sort([rand(1:M) for i in 1:4])
titlelables2 = "Realization " .* string.(process_index)
for i in 1:4
    ax = Axis(fig[2, i]; title=titlelables2[i], options...)
    heatmap!(ax, x[:], y[1:Nd2], θs[process_index[i]][:, 1:Nd2], colormap=:balance, interpolate=true)
    contour!(ax, x[:], y[1:Nd2], θs[process_index[i]][:, 1:Nd2], color=:black, levels=10, linewidth=1.0)
end
display(fig)
