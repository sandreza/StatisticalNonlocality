using StatisticalNonlocality, LinearAlgebra, FFTW, SparseArrays, JLD2
import StatisticalNonlocality: chebyshev, fourier_nodes, fourier_wavenumbers
import StatisticalNonlocality: droprelativezeros!

# transition rate
γ = 1e0  # 1e0 # 1e2
# diffusivity
κ = 1e-2 # 1e0 # 1e-2

# minimal configuration is M = 64, N = 8
N = 8 * 4
M = 8 * 4

# The first step is to build the block operators
a, b = 0, 4π
k = fourier_wavenumbers(N, L = b - a)
x = fourier_nodes(N, a = a, b = b)

ℱ = fft(I + zeros(N, N), 1)
ℱ⁻¹ = ifft(I + zeros(N, N), 1)
Dx = real.(ℱ⁻¹ * Diagonal(im .* k) * ℱ)
Dy = copy(Dx)
x = reshape(x, (N, 1))
y = reshape(x, (1, N))
∂y = kron(Dy, I + zeros(N, N))
∂x = kron(I + zeros(N, N), Dx)
Δ = ∂x^2 + ∂y^2

# convenience functions
advection_operator(ψ, ∂x, ∂y) = Diagonal(∂y * ψ[:]) * ∂x + Diagonal(-∂x * ψ[:]) * ∂y

makelocal(E) = Array(Diagonal(sum(E, dims = 2)[:]))

function localdiffusivity(E, N, M)
    local localdiff = makelocal(E)
    localdiff = [localdiff[i, i] for i = 1:size(E)[1]]
    local localdiff = reshape(localdiff, (N, M ))
    return localdiff
end

function avglocaldiffusivity(E, N, M)
    local localdiff = localdiffusivity(E, N, M)
    local localdiff = sum(localdiff, dims = 1)[:] ./ N # zero'th mode
    return localdiff
end

# the local diffusivity is proportional to 1/γ
function grabdiagonal(A)
    MM = minimum(size(A))
    diagA = zeros(MM)
    for i in 1:MM
        diagA[i] = A[i, i]
    end
    return diagA
end

# random phase wavenumbers
wavemax = 3.0 # change to 3 when automated
𝓀 = collect(-wavemax:0.5:wavemax)
𝓀ˣs = reshape(𝓀, (length(𝓀), 1))
𝓀ʸs = reshape(𝓀, (1, length(𝓀)))
A = @. 0.1 * (𝓀ˣs * 𝓀ˣs + 𝓀ʸs * 𝓀ʸs)^(-11 / 12)
A[A.==Inf] .= 0.0

for (i, 𝓀ˣ) in enumerate(𝓀ˣs), (j, 𝓀ʸ) in enumerate(𝓀ʸs)
    filename = "lots_of_diffusivities_kx_" * string(𝓀ˣ) * "ky_" * string(𝓀ʸ) * ".jld2"
    println("-----------")
    println("currently on wavenumber 𝓀ˣ=", 𝓀ˣ, ", 𝓀ʸ=", 𝓀ʸ)
    println("The amplitude is ", A[i, j])
    println("-----------")
    U₀ = A[i, j]

    ψ¹ = @. U₀ * sin(𝓀ˣ * x + 𝓀ʸ * y)
    ψ² = @. U₀ * cos(𝓀ˣ * x + 𝓀ʸ * y)
    ψ³ = @. -U₀ * sin(𝓀ˣ * x + 𝓀ʸ * y)
    ψ⁴ = @. -U₀ * cos(𝓀ˣ * x + 𝓀ʸ * y)

    A¹ = advection_operator(ψ¹, ∂x, ∂y)
    A² = advection_operator(ψ², ∂x, ∂y)
    A³ = advection_operator(ψ³, ∂x, ∂y)
    A⁴ = advection_operator(ψ⁴, ∂x, ∂y)

    L = [(γ*I-κ*Δ) 0.0*I 0.5*A¹; 0.0*I (γ*I-κ*Δ) -0.5*A²; A¹ -A² (2*γ*I-κ*Δ)]
    G = L \ I

    # Create the block linear operator for components of the diffusivity tensor
    u¹ = Diagonal(∂y * ψ¹[:])
    u² = Diagonal(∂y * ψ²[:])
    v¹ = Diagonal(-∂x * ψ¹[:])
    v² = Diagonal(-∂x * ψ²[:])

    U = -0.5 * [u¹; u²; 0 * I]
    Uᵀ = [u¹ u² 0 * I]
    V = -0.5 * [v¹; v²; 0 * I]
    Vᵀ = [v¹ v² 0 * I]

    EF¹¹ = Uᵀ * G * U # flux in x due to gradients in x
    EF¹² = Uᵀ * G * V # flux in x due to gradients in z
    EF²¹ = Vᵀ * G * U # flux in z due to gradients in x
    EF²² = Vᵀ * G * V # flux in z due to gradients in z

    E = [EF¹¹ EF¹²; EF²¹ EF²²]

    λE = eigvals(E)
    bools = real.(λE) .≤ eps(1e3 * maximum(abs.(λE)))
    println(sum(bools) == length(λE))

    # OUTPUT
    # always overwrite
    data_directory = "data"
    mkpath(data_directory)
    filepath = data_directory * "/" * filename
    if isfile(filepath)
        rm(filepath)
    end
    ##
    file = jldopen(filepath, "a+")
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
    file[groupname]["y"] = y
    # transition matrix 
    groupname = "transition"
    JLD2.Group(file, groupname)

    T = [-1 1 0 0; 0 -1 1 0; 0 0 -1 1; 1 0 0 -1]
    sT = (T + T') / 2

    file[groupname]["T"] = γ * sT
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
    file[groupname]["κ11"] = localdiffusivity(EF¹¹, N, N)
    file[groupname]["κ12"] = localdiffusivity(EF¹², N, N)
    file[groupname]["κ21"] = localdiffusivity(EF²¹, N, N)
    file[groupname]["κ22"] = localdiffusivity(EF²², N, N)

    # zonally averaged local diffusivities
    groupname = "averagelocaldiffusivity"
    JLD2.Group(file, groupname)
    file[groupname]["κ11"] = avglocaldiffusivity(EF¹¹, N, N)
    file[groupname]["κ12"] = avglocaldiffusivity(EF¹², N, N)
    file[groupname]["κ21"] = avglocaldiffusivity(EF²¹, N, N)
    file[groupname]["κ22"] = avglocaldiffusivity(EF²², N, N)

    close(file)

end