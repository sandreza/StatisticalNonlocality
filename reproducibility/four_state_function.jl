using StatisticalNonlocality, LinearAlgebra, FFTW, SparseArrays, JLD2
import StatisticalNonlocality: chebyshev, fourier_nodes, fourier_wavenumbers
import StatisticalNonlocality: droprelativezeros!

function construct_four_state_filename(parameters, M, N, dirichlet)
    (; U, γ, ω, κ) = parameters
    base_name = "four_state_"
    descriptor = String[]
    push!(descriptor, "U_" * string(U) * "_")
    push!(descriptor, "γ_" * string(γ) * "_")
    push!(descriptor, "ω_" * string(ω) * "_")
    push!(descriptor, "κ_" * string(κ) * "_")
    push!(descriptor, "M_" * string(M) * "_")
    push!(descriptor, "N_" * string(N) * "_")
    if dirichlet
        push!(descriptor, "dirichlet" * ".jld2")
    else
        push!(descriptor, "neumann" * ".jld2")
    end
    return base_name * prod(descriptor)
end

# Helper functions

makelocal(E) = Array(Diagonal(sum(E, dims = 2)[:]))

function localdiffusivity(E, N, M)
    local localdiff = makelocal(E)
    localdiff = [localdiff[i, i] for i = 1:size(E)[1]]
    local localdiff = reshape(localdiff, (N, M + 1))
    return localdiff
end

function avglocaldiffusivity(E, N, M)
    local localdiff = localdiffusivity(E, N, M)
    local localdiff = sum(localdiff, dims = 1)[:] ./ N # zero'th mode
    return localdiff
end

function grabdiagonal(A)
    MM = minimum(size(A))
    diagA = zeros(MM)
    for i in 1:MM
        diagA[i] = A[i, i]
    end
    return diagA
end

function advection_operator(ψ, ∂x, ∂z) 
    return Diagonal(∂z * ψ[:]) * ∂x + Diagonal(-∂x * ψ[:]) * ∂z
end

# Main Function
function four_state(parameters; M = 8 * 8, N = 8, filename = nothing, dirichlet = false)
    (; U, γ, ω, κ) = parameters
    U₀ = U
    if isnothing(filename)
        filename = construct_four_state_filename(parameters, M, N, dirichlet)
    end

    Dz, z = chebyshev(M)
    a, b = 0, 2π
    k = fourier_wavenumbers(N, L = b - a)
    x = fourier_nodes(N, a = a, b = b)

    ℱ = fft(I + zeros(N, N), 1)
    ℱ⁻¹ = ifft(I + zeros(N, N), 1)
    Dx = real.(ℱ⁻¹ * Diagonal(im .* k) * ℱ)
    x = reshape(x, (N, 1))
    z = reshape(z, (1, M + 1))

    ψ¹ = U₀ * sin.(x) .* cos.(π / 2 * z)
    ψ² = U₀ * cos.(x) .* cos.(π / 2 * z)
    ψ³ = -U₀ * sin.(x) .* cos.(π / 2 * z)
    ψ⁴ = -U₀ * cos.(x) .* cos.(π / 2 * z)

    ∂z = kron(Dz, I + zeros(N, N))
    ∂x = kron(I + zeros(M + 1, M + 1), Dx)

    zlifted = sparse(kron(Diagonal(z[:]), I + zeros(N, N)))

    A¹ = advection_operator(ψ¹, ∂x, ∂z)
    A² = advection_operator(ψ², ∂x, ∂z)

    Δ = ∂x^2 + ∂z^2

    # Create the block linear operator
    L = [(γ*I-κ*Δ) -ω*I 0.5*A¹; ω*I (γ*I-κ*Δ) -0.5*A²; A¹ -A² (2*γ*I-κ*Δ)]

    # Grab Boundary Indices
    zlifted = sparse(kron(Diagonal(z[:]), I + zeros(N, N)))
    zlifted = [zlifted 0*I 0*I; 0*I zlifted 0*I; 0*I 0*I zlifted]
    ∂Ω¹ = zlifted.rowval[zlifted.nzval.==z[1]]

    # Impose Boundary conditions at z[1]
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

    # Impost boundary conditions at z[end]
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

    # Find the Greens Function
    G = qr(L) \ I

    # Grab Components of the Velocity field
    u¹ = Diagonal(∂z * ψ¹[:])
    u² = Diagonal(∂z * ψ²[:])
    v¹ = Diagonal(-∂x * ψ¹[:])
    v² = Diagonal(-∂x * ψ²[:])
    # Account for Boundary conditions in U and V 
    # Can be thought of as an operator that masks the values on boundaries 
    # Technically the mask is applied to the entire rhs, but since we want the operator 
    # we just applied it here.
    # This is equivalent since diagonal matrices commute and masking is idempotent
    U = -0.5 * [u¹; u²; 0 * I]
    Uᵀ = [u¹ u² 0 * I]
    V = -0.5 * [v¹; v²; 0 * I]
    Vᵀ = [v¹ v² 0 * I]

    ∂Ω = vcat(∂Ω¹, ∂Ω²)
    for ∂i in ∂Ω
        U[∂i, :] .= 0.0
        V[∂i, :] .= 0.0
    end

    EF¹¹ = Uᵀ * G * U # flux in x due to gradients in x
    EF¹² = Uᵀ * G * V # flux in x due to gradients in z
    EF²¹ = Vᵀ * G * U # flux in z due to gradients in x
    EF²² = Vᵀ * G * V # flux in z due to gradients in z

    E = [EF¹¹ EF¹²; EF²¹ EF²²]

    EF¹¹ *= -1.0
    EF¹² *= -1.0
    EF²¹ *= -1.0
    EF²² *= -1.0

    # always overwrite
    data_directory = "data"
    mkpath(data_directory)
    filepath = data_directory * "/" * filename
    if isfile(filepath)
        rm(filepath)
    end

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
    file[groupname]["ω"] = ω
    file[groupname]["κ"] = κ
    file[groupname]["U"] = U₀

    # grid 
    groupname = "grid"
    JLD2.Group(file, groupname)
    file[groupname]["x"] = x
    file[groupname]["z"] = z

    # transition matrix 
    groupname = "transition"
    JLD2.Group(file, groupname)

    T = [-1 1 0 0; 0 -1 1 0; 0 0 -1 1; 1 0 0 -1]
    sT = (T + T') / 2
    aT = (T - T') / 2

    file[groupname]["T"] = γ * sT + ω * aT

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
    file[groupname]["κ11"] = localdiffusivity(EF¹¹, N, M)
    file[groupname]["κ12"] = localdiffusivity(EF¹², N, M)
    file[groupname]["κ21"] = localdiffusivity(EF²¹, N, M)
    file[groupname]["κ22"] = localdiffusivity(EF²², N, M)

    # zonally averaged local diffusivities
    groupname = "averagelocaldiffusivity"
    JLD2.Group(file, groupname)
    file[groupname]["κ11"] = avglocaldiffusivity(EF¹¹, N, M)
    file[groupname]["κ12"] = avglocaldiffusivity(EF¹², N, M)
    file[groupname]["κ21"] = avglocaldiffusivity(EF²¹, N, M)
    file[groupname]["κ22"] = avglocaldiffusivity(EF²², N, M)

    close(file)
    return filename
end