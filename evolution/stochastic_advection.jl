@info "initializing stochastic advection fields"
using StatisticalNonlocality
using FFTW, LinearAlgebra, BenchmarkTools, Random, HDF5, ProgressBars, Statistics
ArrayType = Array

rng = MersenneTwister(12345)
Random.seed!(12)

Ns = (16, 1024 * 2)
Ω = S¹(2π) × S¹(1)

grid = FourierGrid(Ns, Ω, arraytype=ArrayType)
nodes, wavenumbers = grid.nodes, grid.wavenumbers

# build operators
x = nodes[1]
kˣ = wavenumbers[1]
∂x = im * kˣ
Δ = @. ∂x^2
κ = 0.01
𝒟 = κ .* Δ

θ = ArrayType(zeros(ComplexF64, Ns...))
source = ones(Ns[1], 1) .* (Ns[1] / 2)
source[1] = 0 # no mean source
source[floor(Int, Ns[1] / 2)+1] = 0 # no aliased source
u = ArrayType(zeros((1, Ns[2])))
auxiliary = [copy(θ) for i in 1:5]

## Plan FFT 
P = plan_fft!(θ, 1)
P⁻¹ = plan_ifft!(θ, 1)

##
Δx = x[2] - x[1]
umax = 1
cfl = 0.3
dt = cfl * minimum([Δx / umax, Δx^2 / κ])
timetracker = zeros(2)
timetracker[2] = 0.1
stochastic = (; noise_amplitude=1.0, ou_amplitude=1.0)
operators = (; ∂x, 𝒟)

stochastic_auxiliary = copy(u)
parameters = (; operators, source, stochastic)
##
function ou_rhs!(φ̇, φ, parameters)
    γ = parameters.ou_amplitude
    @. φ̇ = -γ * φ
end

function advection_rhs!(Ṡ, S, u, t, parameters)
    (; ∂x, 𝒟) = parameters.operators
    source = parameters.source
    @. Ṡ = -u * ∂x * S + 𝒟 * S + source
end
##
struct StochasticRungeKutta4{S1,S2,F1,F2,P,T}
    auxiliary::S1
    stochastic_auxiliary::S2
    rhs!::F1
    stochastic_rhs!::F2
    parameters::P
    timetracker::T
end

function (step!::StochasticRungeKutta4)(S, φ, rng)
    S̃, k₁, k₂, k₃, k₄ = step!.auxiliary
    Δt = step!.timetracker[2]
    t = view(step!.timetracker, 1)
    rhs! = step!.rhs!
    parameters = step!.parameters

    rhs!(k₁, S, φ, t, parameters)
    @. S̃ = S + Δt * k₁ * 0.5
    t[1] += Δt / 2
    step!(φ, rng)
    rhs!(k₂, S̃, φ, t, parameters)
    @. S̃ = S + Δt * k₂ * 0.5
    rhs!(k₃, S̃, φ, t, parameters)
    @. S̃ = S + Δt * k₃
    t[1] += Δt / 2
    step!(φ, rng)
    rhs!(k₄, S̃, φ, t, parameters)
    @. S += Δt / 6 * (k₁ + 2 * k₂ + 2 * k₃ + k₄)
end

function (stochastic_step!::StochasticRungeKutta4)(φ, rng)
    φ̇ = stochastic_step!.stochastic_auxiliary
    Δt = stochastic_step!.timetracker[2]
    rhs! = stochastic_step!.stochastic_rhs!
    parameters = step!.parameters.stochastic
    ϵ = parameters.noise_amplitude
    rhs!(φ̇, φ, parameters)
    @. φ += φ̇ * Δt / 2
    randn!(rng, φ̇)
    @. φ += ϵ * sqrt(Δt / 2 * 2) * φ̇ # now at t = 0.5, note the factor of two has been accounted for
end
##
step! = StochasticRungeKutta4(auxiliary, stochastic_auxiliary, advection_rhs!, ou_rhs!, parameters, timetracker)
us = Vector{Float64}[]
uθs = Vector{Float64}[]
θs = Vector{Float64}[]
rng = MersenneTwister(12345)
fluxstart = 10000
iterations = 3 * 20000
for i in ProgressBar(1:iterations)
    step!(θ, u, rng)
    if (i > fluxstart) & (i % 1 == 0)
        push!(us, u[:])
        uθ = -imag(mean(u .* θ, dims=2))[:]
        push!(uθs, uθ[:])
        push!(θs, real.(mean(θ, dims=2)[:]))
    end
end

flux = mean(uθs) / Ns[1] * 2
ensemble_mean = mean(θs) / Ns[1] * 2

keff1 = @. (1 / ensemble_mean - κ * kˣ^2) / kˣ^2
keff2 = @. flux / (ensemble_mean * kˣ)
keff1 = keff1[2:floor(Int, Ns[1] / 2)]
keff2 = keff2[2:floor(Int, Ns[1] / 2)]
## Save 
@info "saving data for 1D OU"
hfile = h5open(pwd() * "/data/comparison.hdf5", "w")
hfile["keff1"] = keff1
hfile["keff2"] = keff2
hfile["flux"] = flux
hfile["ensemble_mean"] = ensemble_mean
hfile["kx"] = kˣ
close(hfile)
