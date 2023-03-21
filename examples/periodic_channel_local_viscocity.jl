using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2
using StatisticalNonlocality: ou_transition_matrix, uniform_phase
using MarkovChainHammer.TransitionMatrix: steady_state
using StatisticalNonlocality
using ProgressBars
rng = MersenneTwister(1234)
Random.seed!(123456789)

using CUDA
arraytype = Array
Ω = S¹(2π) × S¹(2)
N = 2^5 # number of gridpoints
grid = FourierGrid(N, Ω, arraytype=arraytype)
nodes, wavenumbers = grid.nodes, grid.wavenumbers
M = 1



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
    pⁱ = p[i]
    θ .= (s ./ (tmp * κ)) .* pⁱ
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
##
u1 = copy(u)
v1 = copy(v)
κxx = 0.0 .+ 1 * local_diffusivity_tensor[:, :, 1, 1] # @. real(1 / (2(γ^2 + ω^2)) * (γ * (u1 * u1 + u2 * u2) + ω * (u1 * u2 - u2 * u1)))
κyy = 0.0 .+ 1 * local_diffusivity_tensor[:, :, 2, 2] # @. real(1 / (2(γ^2 + ω^2)) * (γ * (v1 * v1 + v2 * v2) + ω * (v1 * v2 - v2 * v1)))
κyx = 0.0 .+ 1 * local_diffusivity_tensor[:, :, 2, 1] # @. real(1 / (2(γ^2 + ω^2)) * (γ * (u1 * v1 + u2 * v2) + ω * (u1 * v2 - u2 * v1)))
κxy = 0.0 .+ 1 * local_diffusivity_tensor[:, :, 1, 2] # @. real(1 / (2(γ^2 + ω^2)) * (γ * (u1 * v1 + u2 * v2) + ω * (u2 * v1 - u1 * v2)))

## timestepping
Δx = x[2] - x[1]
cfl = 0.1 #0.1
eddy_diffusive_Δt = cfl * Δx^2 / maximum(abs.(κxx)) 
diffusive_Δt = cfl * Δx^2 / κ
transition_Δt = cfl / maximum(-real.(eigvals(Q)))
Δt = min(eddy_diffusive_Δt, diffusive_Δt, transition_Δt)

simulation_parameters = (; κxx, κxy, κyx, κyy, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ)

function n_state_rhs_local!(θ̇s, θs, simulation_parameters)
    (; κxx, κxy, κyx, κyy, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ) = simulation_parameters
    θ̇ = θ̇s[1]
    θ = θs[1]
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
    @. uθ = κxx * ∂ˣθ + κxy * ∂ʸθ
    @. vθ = κyx * ∂ˣθ + κyy * ∂ʸθ
    P * uθ
    P * vθ
    @. ∂ˣuθ = ∂x * uθ
    @. ∂ʸvθ = ∂y * vθ
    # go back to real space 
    [P⁻¹ * field for field in (∂ˣuθ, ∂ʸvθ)] # in place ifft
    # compute θ̇ in real space
    @. θ̇ = ∂ˣuθ + ∂ʸvθ + κΔθ + s
    return nothing
end

n_state_rhs_local!(θ̇s, θs, simulation_parameters)
##
rhs! = n_state_rhs_local!

tend = 10.0
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
Nd2 = floor(Int, N / 2) + 1
fig = Figure(resolution=(2100, 1000))
titlelables1 = ["Diffusivity Tensor"]
options = (; titlesize=30, xlabel="x", ylabel="y", xlabelsize=40, ylabelsize=40, xticklabelsize=30, yticklabelsize=30)
for i in 1:M
    ax = Axis(fig[1, i]; title=titlelables1[i], options...)
    heatmap!(ax, x[:], y[1:Nd2], real.(θs[i])[:, 1:Nd2], colormap=:balance, interpolate=true)
    contour!(ax, x[:], y[1:Nd2], real.(θs[i])[:, 1:Nd2], color=:black, levels=10, linewidth=1.0)
end
display(fig)
diffθ =  real.(copy(θs[1]))