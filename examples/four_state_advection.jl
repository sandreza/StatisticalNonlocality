using StatisticalNonlocality, LinearAlgebra, FFTW, SparseArrays, JLD2
import StatisticalNonlocality: chebyshev, fourier_nodes, fourier_wavenumbers
import StatisticalNonlocality: droprelativezeros!
using CUDA
array_type = Array

# minimal configuration is M = 64, N = 8
N = 8 * 4
M = 8 * 4 

# The first step is to build the block operators
a, b = 0, 4π
k = fourier_wavenumbers(N, L=b - a)
x = fourier_nodes(N, a=a, b=b)
Dz, z = chebyshev(M)
ℱ = fft(I + zeros(N, N), 1)
ℱ⁻¹ = ifft(I + zeros(N, N), 1)
Dx = real.(ℱ⁻¹ * Diagonal(im .* k) * ℱ)

function build_operator_dirichlet(Dz, k; array_type = Array, μ = 0.0)
    A = Dz * Dz .- k^2 .- μ
    A[1, :] .= 0.0
    A[1, 1] = 1.0
    A[end, :] .= 0
    A[end, end] = 1.0
    return lu(array_type(A))
end
function build_operator_neuman(Dz, k; array_type = Array, μ = 0.0)
    A = Dz * Dz .- k^2 .- μ
    A[1, :] .= Dz[1,:]
    A[end, :] = Dz[end, :]
    return lu(array_type(A))
end

x = array_type(reshape(x, (1, N)))
z = array_type(reshape(z, (M + 1, 1)))
U₀ = 1.0
ψ¹ = U₀ * sin.(x) .* cos.(π / 2 * z)
ψ² = U₀ * cos.(x) .* cos.(π / 2 * z)
ψ³ = -U₀ * sin.(x) .* cos.(π / 2 * z)
ψ⁴ = -U₀ * cos.(x) .* cos.(π / 2 * z)

u¹ = Dz * ψ¹
u² = Dz * ψ²
v¹ = - ψ¹ * (Dx')
v² = -ψ² * (Dx')
# check incompressibility and derivatives
maximum(abs.(u¹ * (Dx') + Dz * v¹))
maximum(abs.(u² * (Dx') + Dz * v²))
∂x = reshape(im .* k, (1, 32))
t¹ = -real.(ifft(∂x .* fft(ψ¹)))
t¹ - v¹
##
lu_fact = [build_operator_neuman(Dz, k[i]; array_type=array_type) for i in 1:round(Int, N / 2)]
rhs = ones(size(Dz)[1], 100)
rhs[1, :] .= 0.0
rhs[end, :] .= 0.0
rhs = array_type(rhs)
sol = lu_fact[2] \ rhs
##
# Need to evolve ∂ₜ θ + u⋅∇θ = κ∇²θ + s 
# use update rule in Fourier Space    (-Δt*κ∇² + 1) θⁿ⁺¹ = θⁿ + Δt(∇⋅(uθⁿ) + sⁿ)
# or  (-∇² + 1/(Δt*κ)) θⁿ⁺¹ = θⁿ/(Δt*κ) + (∇⋅(uθⁿ) + sⁿ)/κ
ldiv!(Y, A, B)

function ensemble_rhs!(θ̇s, θs, parameters)
    (; A, u, v, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ) = simulation_parameters

    
end