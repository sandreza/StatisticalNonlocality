using LinearAlgebra
T = [-1 1 0 0; 0 -1 1 0; 0 0 -1 1; 1 0 0 -1]

Λ, V = eigen(T)
Λ, W = eigen(T')

# Right eigenvectors
V⁴ = round.(V[:, 1] ./ V[1, 1])
V³ = round.(V[:, 2] ./ V[1, 2])
V² = round.(V[:, 3] ./ V[1, 3])
V¹ = round.(V[:, 4] ./ V[1, 4])

# Left eigenvectors
W⁴ = round.(W[:, 1] ./ W[1, 1])
W³ = round.(W[:, 2] ./ W[1, 2])
W² = round.(W[:, 3] ./ W[1, 3])
W¹ = round.(W[:, 4] ./ W[1, 4])

# Flip Order (take into account later)
W[:, 1] .= W¹
W[:, 2] .= W²
W[:, 3] .= W³
W[:, 4] .= W⁴
inv(W) * 4

# Quick Check (right eigenvector)
T * V - V * Diagonal(Λ)
maximum(abs.(T * V - V * Diagonal(Λ))) ≤ eps(10.0)
# Quick Check (left eigenvector)
T' * W - W * Diagonal(reverse(Λ))    # since we flipped the order so that index 1 corresponds to the lowest eigenvalue
W' * T - Diagonal(reverse(Λ))' * W'  # same thing, recall that ' means adjoint, thus takes the conjucate
maximum(abs.(T' * W - W * Diagonal(reverse(Λ)))) ≤ eps(10.0)

# The following transformation is what is used
S = real.(V)
S[:, 1] .= real.(W¹)
S[:, 2] .= real.(0.5 * (W² + W³))
S[:, 3] .= real.(0.5 * im * (W² - W³))
S[:, 4] .= real.(W⁴)


##
using StatisticalNonlocality, LinearAlgebra, FFTW, SparseArrays
import StatisticalNonlocality: chebyshev, fourier_nodes, fourier_wavenumbers
import StatisticalNonlocality: droprelativezeros!
N = 8 * 2
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

γ = 1e2
κ = 1e-2
L = [(γ*I-κ*Δ) -γ*I 0.5*A¹; γ*I (γ*I-κ*Δ) -0.5*A²; A¹ -A² (2*γ*I-κ*Δ)]

# Choose BC 
dirichlet = false
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
##

using GLMakie
fig = Figure(resolution = (1800, 1300), title = "Nonlocal Operators")
titlestring = "Kˣˣ"
ax1 = Axis(fig[1, 1], title = titlestring, titlesize = 30)
titlestring = "Kˣᶻ"
ax2 = Axis(fig[2, 1], title = titlestring, titlesize = 30)
titlestring = "Kᶻˣ"
ax3 = Axis(fig[1, 3], title = titlestring, titlesize = 30)
titlestring = "Kᶻᶻ"
ax4 = Axis(fig[2, 3], title = titlestring, titlesize = 30)

colormap = :thermal
colormap2 = :balance
hm1 = heatmap!(ax1, EF¹¹, colormap = colormap)
ax1.yreversed = true

hm2 = heatmap!(ax2, EF¹², colormap = colormap2)
ax2.yreversed = true

hm3 = heatmap!(ax3, EF²¹, colormap = colormap2)
ax3.yreversed = true

hm4 = heatmap!(ax4, EF²², colormap = colormap)
ax4.yreversed = true

Colorbar(fig[1, 2], hm1, height = Relative(3 / 4), width = 25, ticklabelsize = 30,
    labelsize = 30, ticksize = 25, tickalign = 1,)
Colorbar(fig[2, 2], hm2, height = Relative(3 / 4), width = 25, ticklabelsize = 30,
    labelsize = 30, ticksize = 25, tickalign = 1,)
Colorbar(fig[1, 4], hm3, height = Relative(3 / 4), width = 25, ticklabelsize = 30,
    labelsize = 30, ticksize = 25, tickalign = 1,)
Colorbar(fig[2, 4], hm4, height = Relative(3 / 4), width = 25, ticklabelsize = 30,
    labelsize = 30, ticksize = 25, tickalign = 1,)
display(fig)
##
# Local
fig = Figure(resolution = (1800, 1300), title = "Local Operators")
titlestring = "Kˣˣ"
ax1 = Axis(fig[1, 1], title = titlestring, titlesize = 30)
titlestring = "Kˣᶻ"
ax2 = Axis(fig[2, 1], title = titlestring, titlesize = 30)
titlestring = "Kᶻˣ"
ax3 = Axis(fig[1, 3], title = titlestring, titlesize = 30)
titlestring = "Kᶻᶻ"
ax4 = Axis(fig[2, 3], title = titlestring, titlesize = 30)

colormap = :thermal
colormap2 = :balance
hm1 = heatmap!(ax1, makelocal(EF¹¹), colormap = colormap)
ax1.yreversed = true

hm2 = heatmap!(ax2, makelocal(EF¹²), colormap = colormap2)
ax2.yreversed = true

hm3 = heatmap!(ax3, makelocal(EF²¹), colormap = colormap2)
ax3.yreversed = true

hm4 = heatmap!(ax4, makelocal(EF²²), colormap = colormap)
ax4.yreversed = true

Colorbar(fig[1, 2], hm1, height = Relative(3 / 4), width = 25, ticklabelsize = 30,
    labelsize = 30, ticksize = 25, tickalign = 1,)
Colorbar(fig[2, 2], hm2, height = Relative(3 / 4), width = 25, ticklabelsize = 30,
    labelsize = 30, ticksize = 25, tickalign = 1,)
Colorbar(fig[1, 4], hm3, height = Relative(3 / 4), width = 25, ticklabelsize = 30,
    labelsize = 30, ticksize = 25, tickalign = 1,)
Colorbar(fig[2, 4], hm4, height = Relative(3 / 4), width = 25, ticklabelsize = 30,
    labelsize = 30, ticksize = 25, tickalign = 1,)
display(fig)

##

function avglocaldiffusivity(E, N, M)
    local localdiff = makelocal(E)
    localdiff = [localdiff[i, i] for i = 1:size(E)[1]]
    local localdiff = reshape(localdiff, (N, M + 1))
    local localdiff = sum(localdiff, dims = 1)[:]
    return localdiff
end

localdiff = avglocaldiffusivity(EF¹¹, N, M)
lines(localdiff, z[:])
##
fig = Figure(resolution = (1800, 1300), title = "Local Operators")
titlestring = "Kˣˣ"
ax1 = Axis(fig[1, 1], title = titlestring, titlesize = 30)
titlestring = "Kˣᶻ"
ax2 = Axis(fig[2, 1], title = titlestring, titlesize = 30)
titlestring = "Kᶻˣ"
ax3 = Axis(fig[1, 2], title = titlestring, titlesize = 30)
titlestring = "Kᶻᶻ"
ax4 = Axis(fig[2, 2], title = titlestring, titlesize = 30)

colormap = :thermal
colormap2 = :balance
hm1 = lines!(ax1, avglocaldiffusivity(EF¹¹, N, M), z[:])
hm2 = lines!(ax2, avglocaldiffusivity(EF¹², N, M), z[:])
hm3 = lines!(ax3, avglocaldiffusivity(EF²¹, N, M), z[:])
hm4 = lines!(ax4, avglocaldiffusivity(EF²², N, M), z[:])

display(fig)

##
# Stream Functions 
fig = Figure(resolution = (1800, 1300), title = "Local Operators")
titlestring = "ψ¹"
ax1 = Axis(fig[1, 1], title = titlestring, titlesize = 30)
titlestring = "ψ²"
ax2 = Axis(fig[1, 3], title = titlestring, titlesize = 30)
titlestring = "ψ³"
ax3 = Axis(fig[2, 1], title = titlestring, titlesize = 30)
titlestring = "ψ⁴"
ax4 = Axis(fig[2, 3], title = titlestring, titlesize = 30)

colormap = :balance
hm1 = heatmap!(ax1, ψ¹, colormap = colormap, interpolate = true)

hm2 = heatmap!(ax2, ψ², colormap = colormap, interpolate = true)

hm3 = heatmap!(ax3, ψ³, colormap = colormap, interpolate = true)

hm4 = heatmap!(ax4, ψ⁴, colormap = colormap, interpolate = true)

Colorbar(fig[1, 2], hm1, height = Relative(3 / 4), width = 25, ticklabelsize = 30,
    labelsize = 30, ticksize = 25, tickalign = 1,)
Colorbar(fig[2, 2], hm2, height = Relative(3 / 4), width = 25, ticklabelsize = 30,
    labelsize = 30, ticksize = 25, tickalign = 1,)
Colorbar(fig[1, 4], hm3, height = Relative(3 / 4), width = 25, ticklabelsize = 30,
    labelsize = 30, ticksize = 25, tickalign = 1,)
Colorbar(fig[2, 4], hm4, height = Relative(3 / 4), width = 25, ticklabelsize = 30,
    labelsize = 30, ticksize = 25, tickalign = 1,)
display(fig)