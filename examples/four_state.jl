using LinearAlgebra

# overall velocity scale
U₀ = 1.0
# transition rate
γ = 1e2  # 1e0 # 1e2
# diffusivity
κ = 1e-2 # 1e0 # 1e-2
# boundary condition
dirichlet = false
# output file name
filename = "nearly_local.jld2"

using StatisticalNonlocality, LinearAlgebra, FFTW, SparseArrays
import StatisticalNonlocality: chebyshev, fourier_nodes, fourier_wavenumbers
import StatisticalNonlocality: droprelativezeros!
N = 8 * 1
M = 8 * 4
Dz, z = chebyshev(M)
a, b = 0, 2π
k = fourier_wavenumbers(N, L = b - a)
x = fourier_nodes(N, a = a, b = b)

ℱ = fft(I + zeros(N, N), 1)
ℱ⁻¹ = ifft(I + zeros(N, N), 1)
Dx = real.(ℱ⁻¹ * Diagonal(im .* k) * ℱ)
x = reshape(x, (N, 1))
z = reshape(z, (1, M + 1))

ψ¹ = sin.(x) .* cos.(π / 2 * z)
ψ² = cos.(x) .* cos.(π / 2 * z)
ψ³ = -sin.(x) .* cos.(π / 2 * z)
ψ⁴ = -cos.(x) .* cos.(π / 2 * z)

∂z = kron(Dz, I + zeros(N, N))
∂x = kron(I + zeros(M + 1, M + 1), Dx)

e1 = norm(∂x * ψ¹[:] - ψ²[:])
println("The error in take in the partial with respect to x is ", e1)
e2 = norm(∂z * (ψ¹[:]) - (sin.(x).*cos.(z))[:])
println("The error in taking the partial with respect to y is ", e2)

zlifted = sparse(kron(Diagonal(z[:]), I + zeros(N, N)))
zlifted2 = sparse(kron(I + zeros(N, N), Diagonal(z[:])))

##
# u⃗⋅∇
advection_operator(ψ, ∂x, ∂z) =
    Diagonal(∂z * ψ[:]) * ∂x + Diagonal(-∂x * ψ[:]) * ∂z

A¹ = advection_operator(ψ¹, ∂x, ∂z)
A² = advection_operator(ψ², ∂x, ∂z)
A³ = advection_operator(ψ³, ∂x, ∂z)
A⁴ = advection_operator(ψ⁴, ∂x, ∂z)

Δ = ∂x^2 + ∂z^2

tmp = sparse(A⁴)
println(length(tmp.nzval))
droprelativezeros!(tmp)
println(length(tmp.nzval))


L = [(γ*I-κ*Δ) -γ*I 0.5*A¹; γ*I (γ*I-κ*Δ) -0.5*A²; A¹ -A² (2*γ*I-κ*Δ)]

# Grab Boundary Indices
zlifted = sparse(kron(Diagonal(z[:]), I + zeros(N, N)))
zlifted = [zlifted 0*I 0*I; 0*I zlifted 0*I; 0*I 0*I zlifted]
∂Ω¹ = zlifted.rowval[zlifted.nzval.==z[1]]
for (i, ∂i) in enumerate(∂Ω¹)
    if dirichlet
        L[∂i, :] .= 0.0
        L[∂i, ∂i] = 1.0
    else
        L[∂i, :] .= 0.0
        blockindex = div((i - 1), N)
        inds = (blockindex*N*(M+1)+1):((blockindex+1)*N*(M+1))
        L[∂i, inds] .= ∂z[∂Ω¹[(i-1)%N+1], :]
    end
end
∂Ω² = zlifted.rowval[zlifted.nzval.==z[end]]
for (i, ∂i) in enumerate(∂Ω²)
    if dirichlet
        L[∂i, :] .= 0.0
        L[∂i, ∂i] = 1.0
    else
        L[∂i, :] .= 0.0
        blockindex = div((i - 1), N)
        inds = (blockindex*N*(M+1)+1):((blockindex+1)*N*(M+1))
        L[∂i, inds] .= ∂z[∂Ω²[(i-1)%N+1], :]
    end
end
∂Ω = vcat(∂Ω¹, ∂Ω²)

Q, R = qr(L)

G = qr(L) \ I
u¹ = Diagonal(∂z * ψ¹[:])
u² = Diagonal(∂z * ψ²[:])
v¹ = Diagonal(-∂x * ψ¹[:])
v² = Diagonal(-∂x * ψ²[:])
# NEED TO ACCOUNT FOR BCS in U, V
U = -0.5 * [u¹; u²; 0 * I]
Uᵀ = [u¹ u² 0 * I]
V = -0.5 * [v¹; v²; 0 * I]
Vᵀ = [v¹ v² 0 * I]
for ∂i in ∂Ω
    U[∂i, :] .= 0.0
    V[∂i, :] .= 0.0
end

EF¹¹ = Uᵀ * G * U
EF¹² = Uᵀ * G * V
EF²¹ = Vᵀ * G * U
EF²² = Vᵀ * G * V

E = [EF¹¹ EF¹²; EF²¹ EF²²]
# Check that all the eigenvalues are negative
λE = eigvals(E)
bools = real.(λE) .≤ eps(1e3 * maximum(abs.(λE)))
sum(bools) == length(λE)

maximum(abs.(EF²¹ + EF¹²))
EF¹¹ *= -1.0
EF¹² *= -1.0
EF²¹ *= -1.0
EF²² *= -1.0

makelocal(E) = Array(Diagonal(sum(E, dims = 2)[:]))
function avglocaldiffusivity(E, N, M)
    local localdiff = makelocal(E)
    localdiff = [localdiff[i, i] for i = 1:size(E)[1]]
    local localdiff = reshape(localdiff, (N, M + 1))
    local localdiff = sum(localdiff, dims = 1)[:] ./ N # zero'th mode
    return localdiff
end

##
a1 = avglocaldiffusivity(EF¹¹, N, M)
a2 = avglocaldiffusivity(EF¹², N, M)
a3 = avglocaldiffusivity(EF²¹, N, M)
a4 = avglocaldiffusivity(EF²², N, M)

k = 1
ℓ = π / 2
maximum(a1) / maximum(a4) - ℓ^2 / k^2

λdω = maximum(a1) / maximum(a2) * k / ℓ # = λ / ω
maximum(a1) # = 0.5 U₀² * λ / (λ² + ω²) = 0.5 U₀² * 1 / (λ²/ω² + 1) * 1/ω²
maximum(a2) # = 0.5 U₀² * ω / (λ² + ω²) * k / ℓ
maximum(a3) # should be about the same as maximum(a2)
maximum(a4) # = 0.5 U₀² * λ / (λ² + ω²) * (k / ℓ)^2

U₀ = sqrt(norm(u¹[:])^2 + norm(v¹[:])^2) / (length(u¹))^0.5 # more legit considers quadrature stuff
ω = 1 / (λdω^2 + 1) * k / ℓ / maximum(a2) * 0.5 * U₀^2
λ = ω * λdω
a1max = λ / (λ^2 + ω^2) * 0.5 * U₀^2
a2max = ω / (λ^2 + ω^2) * k / ℓ * 0.5 * U₀^2
a4max = λ / (λ^2 + ω^2) * (k / ℓ)^2 * 0.5 * U₀^2

# the local diffusivity is proportional to 1/γ
function grabdiagonal(A)
    MM = minimum(size(A))
    diagA = zeros(MM)
    for i in 1:MM
        diagA[i] = A[i, i]
    end
    return diagA
end

##
using JLD2
file = jldopen("data/" * filename, "a+")
# diffusivity
groupname = "diffusivity"
JLD2.Group(file, groupname)
file[groupname]["K11"] = EF¹¹
file[groupname]["K12"] = EF¹²
file[groupname]["K21"] = EF²¹
file[groupname]["K22"] = EF²²
# parameters
groupname = "parameters"
JLD2.Group(file, groupname)
file[groupname]["γ"] = γ
file[groupname]["κ"] = κ
# grid 
groupname = "grid"
JLD2.Group(file, groupname)
file[groupname]["x"] = x
file[groupname]["z"] = z
# transition matrix 
groupname = "transition"
JLD2.Group(file, groupname)
file[groupname]["T"] = T
# streamfunction
groupname = "streamfunction"
JLD2.Group(file, groupname)
file[groupname]["ψ¹"] = ψ¹
file[groupname]["ψ²"] = ψ²
file[groupname]["ψ³"] = ψ³
file[groupname]["ψ⁴"] = ψ⁴
# velocities
groupname = "velocities"
JLD2.Group(file, groupname)
file[groupname]["u¹"] = reshape(grabdiagonal(u¹), size(ψ¹))
file[groupname]["u²"] = reshape(grabdiagonal(u²), size(ψ¹))
file[groupname]["v¹"] = reshape(grabdiagonal(v¹), size(ψ¹))
file[groupname]["v²"] = reshape(grabdiagonal(v²), size(ψ¹))
# local diffusivities
groupname = "localdiffusivity"
JLD2.Group(file, groupname)
file[groupname]["κ11"] = avglocaldiffusivity(EF¹¹, N, M)
file[groupname]["κ12"] = avglocaldiffusivity(EF¹², N, M)
file[groupname]["κ21"] = avglocaldiffusivity(EF²¹, N, M)
file[groupname]["κ22"] = avglocaldiffusivity(EF²², N, M)

close(file)