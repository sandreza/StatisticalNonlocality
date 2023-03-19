using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2
using StatisticalNonlocality: ou_transition_matrix
using MarkovChainHammer.TransitionMatrix: steady_state
using StatisticalNonlocality
using ProgressBars
rng = MersenneTwister(1234)
Random.seed!(123456789)

using CUDA
arraytype = Array
Ω = S¹(2π)^2
N = 2^7 # number of gridpoints
M = 3 # number of states
U = 1.0 # amplitude factor
Q = ou_transition_matrix(M - 1)
A = U .* collect(range(-sqrt(M - 1), sqrt(M - 1), length=M))
p = steady_state(Q)
Λ, V = eigen(Q)
V[:, end] .= p
for i in 1:M-1
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
u = similar(ψ)
v = similar(ψ)

# theta
θs = [similar(ψ) for i in 1:M]
∂ˣθ = similar(ψ)
∂ʸθ = similar(ψ)
κΔθ = similar(ψ)
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
index = 2
@. s = sin(kˣ[index] * x) * sin(kʸ[index] * y) # could also set source term to zero

# operators
∂x = im * kˣ
∂y = im * kʸ
Δ = @. ∂x^2 + ∂y^2
κ = 0.01

# set equal to diffusive solution 
tmp = (kˣ[index]^2 + kʸ[index]^2)
for (i, θ) in enumerate(θs)
    pⁱ = p[i]
    θ .= (s ./ (tmp * κ)) .* pⁱ
end

println("maximum value of theta before ", maximum(real.(sum(θs))))


# plan ffts
P = plan_fft!(ψ)
P⁻¹ = plan_ifft!(ψ)

# set stream function and hence velocity
@. ψ = cos(kˣ[2] * x) * cos(kʸ[2] * y)
P * ψ # in place fft
# ∇ᵖψ
@. u = -1.0 * (∂y * ψ);
@. v = (∂x * ψ);
P⁻¹ * ψ;
P⁻¹ * u;
P⁻¹ * v; # don't need ψ anymore

us = [u * a for a in A]
vs = [v * a for a in A]
## timestepping
Δx = x[2] - x[1]
cfl = 0.1 #0.1
advective_Δt = cfl * Δx / maximum(real.(u))
diffusive_Δt = cfl * Δx^2 / κ
Δt = min(advective_Δt, diffusive_Δt)

simulation_parameters = (; A, p, Q, u, v, us, vs, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ)

function n_state_rhs_symmetric!(θ̇s, θs, simulation_parameters)
    (; A, p, Q, us, vs, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ) = simulation_parameters

    # need A (amplitude), p (probability of being in state), Q (transition probability)
    for (i, θ) in enumerate(θs)
        Aⁱ = 1.0 # A[i]
        u = us[i]
        v = vs[i]
        pⁱ = p[i]
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
        @. θ̇ = -Aⁱ * (u * ∂ˣθ + v * ∂ʸθ + ∂ˣuθ + ∂ʸvθ) * 0.5 + κΔθ + s * pⁱ
        # transitions
        for (j, θ2) in enumerate(θs)
            Qⁱʲ = Q[i, j]
            θ̇ .+= Qⁱʲ * θ2
        end
    end

    return nothing
end

n_state_rhs_symmetric!(θ̇s, θs, simulation_parameters)

rhs! = n_state_rhs_symmetric!

tend = 2.0
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

println("maximum value of theta after ", maximum(real.(sum(θs))))
##
using GLMakie
fig = Figure()
for i in 1:M
    ax = Axis(fig[1, i], xlabel = "x", ylabel = "y")
    heatmap!(ax, real.(θs[i]), colormap = :balance, interpolate = true)
    contour!(ax, real.(θs[i]), color = :black, levels = 10, linewidth = 1.0)
end
for i in 1:M 
    tmp = copy(θs[1]) * V⁻¹[i, 1]
    for j in 2:M 
        tmp .+= θs[j] * V⁻¹[i, j]
    end
    ax = Axis(fig[2, i], xlabel="x", ylabel="y")
    heatmap!(ax, real.(tmp), colormap=:balance, interpolate=true)
    contour!(ax, real.(tmp), color=:black, levels=10, linewidth=1.0)
end
display(fig)

##
tmp3 = sum(θs)