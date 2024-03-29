function nstate_channel(N, M; U=1.0, ϵ = π / sqrt(8), c= -π / 2, source_index=3, arraytype = Array)
    ## Grid
    Ω = S¹(2π) × S¹(2)
    grid = FourierGrid(N, Ω, arraytype=arraytype)
    nodes, wavenumbers = grid.nodes, grid.wavenumbers
    x = nodes[1]
    y = nodes[2]
    kˣ = wavenumbers[1]
    kʸ = wavenumbers[2]

    ## Operators
    ∂x = im * kˣ
    ∂y = im * kʸ
    Δ = @. ∂x^2 + ∂y^2
    κ = 0.01

    ## Transition Matrix
    Δx = 2π / M
    φs = collect(0:M-1) * Δx
    A = advection_matrix_central(M; Δx)
    𝒟 = discrete_laplacian_periodic(M; Δx)
    Q = -A * c + 𝒟 * ϵ^2
    p = steady_state(Q)

    ## Allocate Arrays
    (; ψ, ψ2, θs, θv, ∂ˣθ, ∂ʸθ, κΔθ, θ̇s, s, θ̅, k₁, k₂, k₃, k₄, θ̃, uθ, vθ, ∂ˣuθ, ∂ʸvθ, us, vs) = allocate_2D_fields(N, M; arraytype=arraytype)

    ## plan ffts
    P = plan_fft!(ψ)
    P⁻¹ = plan_ifft!(ψ)

    ## Allocate Fields with specific values
    [us[i] .= -kʸ[2] * U * cos.(kˣ[2] * x .+ φs[i]) .* cos.(kʸ[2] .* y) for i in 1:M]
    [vs[i] .= -kˣ[2] * U * sin.(kˣ[2] * x .+ φs[i]) .* sin.(kʸ[2] .* y) for i in 1:M]
    @. s = sin(kˣ[2] * x) * sin(kʸ[source_index] * y) + cos(kˣ[3] * x) * sin(kʸ[source_index] * y)

    ## Set Initial Theta Equal to Diffusive Solution
    tmp = (kˣ[source_index]^2 + kʸ[source_index]^2)
    for (i, θ) in enumerate(θs)
        pⁱ = p[i]
        θ .= (s ./ (tmp * κ)) .* pⁱ
    end

    ## Local Diffusivity Tensor
    Q⁺ = -pinv(Q) # time-integrated autocorrelation
    local_diffusivity_tensor = zeros(size(us[1])..., 2, 2)
    for i in 1:M, j in 1:M
        local_diffusivity_tensor[:, :, 1, 1] .+= us[i] .* us[j] * Q⁺[i, j] * p[j]
        local_diffusivity_tensor[:, :, 1, 2] .+= vs[i] .* us[j] * Q⁺[i, j] * p[j]
        local_diffusivity_tensor[:, :, 2, 1] .+= us[i] .* vs[j] * Q⁺[i, j] * p[j]
        local_diffusivity_tensor[:, :, 2, 2] .+= vs[i] .* vs[j] * Q⁺[i, j] * p[j]
    end
    ## Timestepping 
    Δx = x[2] - x[1]
    cfl = 1.0
    advective_u_Δt = cfl * Δx / maximum([maximum(real.(abs.(us[i]))) for i in 1:M])
    advective_v_Δt = cfl * Δx / maximum([maximum(real.(abs.(vs[i]))) for i in 1:M])
    diffusive_Δt = cfl * Δx^2 / κ
    transition_Δt = cfl / maximum(-real.(eigvals(Q)))
    Δt = min(advective_u_Δt, advective_v_Δt, diffusive_Δt, transition_Δt)

    return (; θs, p, Q, us, vs, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ, θ̇s, θ̅, k₁, k₂, k₃, k₄, θ̃, local_diffusivity_tensor, Δt)
end

function nstate_ou(N, M; γ=1.0, ϵ=√2, arraytype=Array, nstate = M)
    ## Grid
    Ω = S¹(2π)^2
    grid = FourierGrid(N, Ω, arraytype=arraytype)
    nodes, wavenumbers = grid.nodes, grid.wavenumbers
    x = nodes[1]
    y = nodes[2]
    kˣ = wavenumbers[1]
    kʸ = wavenumbers[2]

    ## Operators
    ∂x = im * kˣ
    ∂y = im * kʸ
    Δ = @. ∂x^2 + ∂y^2
    κ = 0.01


    ## Transition Matrix
    Δx = 2 / √(nstate - 1)
    uₘ = 1 / sqrt(γ * 2 / ϵ^2) * [Δx * (i - (nstate - 1) / 2) for i in 0:nstate-1] * U
    Q = ou_transition_matrix(nstate - 1) * γ
    p = steady_state(Q)

    ## Allocate Arrays
    (; ψ, ψ2, θs, θv, ∂ˣθ, ∂ʸθ, κΔθ, θ̇s, s, θ̅, k₁, k₂, k₃, k₄, θ̃, uθ, vθ, ∂ˣuθ, ∂ʸvθ, us, vs) = allocate_2D_fields(N, M; arraytype=arraytype)

    ## plan ffts
    P = plan_fft!(ψ)
    P⁻¹ = plan_ifft!(ψ)

    ## Allocate Fields with specific values 
    # ψ = cos.(kˣ[2] * x) .* cos.(kʸ[2] .* y)
    if M == nstate
        for i in 1:M
            us[i] .= uₘ[i] * kʸ[2] * cos.(kˣ[2] * x) .* sin.(kʸ[2] .* y)
            vs[i] .= -uₘ[i] * kˣ[2] * sin.(kˣ[2] * x) .* cos.(kʸ[2] .* y)
        end
    else 
        for i in 1:M
            uval = rand(uₘ)
            us[i] .= uval * kʸ[2] * cos.(kˣ[2] * x) .* sin.(kʸ[2] .* y)
            vs[i] .= -uval * kˣ[2] * sin.(kˣ[2] * x) .* cos.(kʸ[2] .* y)
        end
    end
    @. s = sin(kˣ[2] * x) * sin(kʸ[2] * y) 

    ## Set Initial Theta Equal to Diffusive Solution
    tmp = (kˣ[2]^2 + kʸ[2]^2)
    for (i, θ) in enumerate(θs)
        if M == nstate
            pⁱ = p[i]
        else
            pⁱ = 1.0
        end
        θ .= (s ./ (tmp * κ)) .* pⁱ
    end

    ## Local Diffusivity Tensor
    Q⁺ = -pinv(Q) # time-integrated autocorrelation
    local_diffusivity_tensor = zeros(size(us[1])..., 2, 2)
    if M == nstate
        for i in 1:M, j in 1:M
            local_diffusivity_tensor[:, :, 1, 1] .+= us[i] .* us[j] * Q⁺[i, j] * p[j]
            local_diffusivity_tensor[:, :, 1, 2] .+= vs[i] .* us[j] * Q⁺[i, j] * p[j]
            local_diffusivity_tensor[:, :, 2, 1] .+= us[i] .* vs[j] * Q⁺[i, j] * p[j]
            local_diffusivity_tensor[:, :, 2, 2] .+= vs[i] .* vs[j] * Q⁺[i, j] * p[j]
        end
    end
    ## Timestepping 
    Δx = x[2] - x[1]
    cfl = 1.0
    advective_u_Δt = cfl * Δx / maximum([maximum(real.(abs.(us[i]))) for i in 1:M])
    advective_v_Δt = cfl * Δx / maximum([maximum(real.(abs.(vs[i]))) for i in 1:M])
    diffusive_Δt = cfl * Δx^2 / κ
    transition_Δt = cfl / maximum(-real.(eigvals(Q)))
    Δt = min(advective_u_Δt, advective_v_Δt, diffusive_Δt, transition_Δt)

    return (; ψ, uₘ, kˣ, kʸ, x, y, θs, p, Q, us, vs, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ, θ̇s, θ̅, k₁, k₂, k₃, k₄, θ̃, local_diffusivity_tensor, Δt)
end