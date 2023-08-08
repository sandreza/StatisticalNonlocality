function allocate_fields(N, M; arraytype = Array)
    # capacity
    c⁰ = arraytype(zeros(ComplexF64, N))
    # theta
    θs = [similar(c⁰) for i in 1:M]
    ∂ˣθ = similar(c⁰)
    κΔθ = similar(c⁰)
    θ̇s = [similar(c⁰) .* 0 for i in 1:M]
    return (; c⁰, θs,  ∂ˣθ, κΔθ, θ̇s)
end

# For conditional mean equations 
function rhs!(θ̇s, θs, simulation_parameters)
    (; p, Q, us, c⁰, ∂ˣθ, c⁰, P, P⁻¹, ∂x, κ, Δ, κΔθ, λ) = simulation_parameters
    for (i, θ) in enumerate(θs)
        u = us[i]
        pⁱ = p[i]
        θ̇ = θ̇s[i]
        # dynamics
        P * θ # in place fft
        # ∇θ
        @. ∂ˣθ = ∂x * θ
        # κΔθ
        @. κΔθ = κ * Δ * θ
        # go back to real space 
        [P⁻¹ * field for field in (θ, ∂ˣθ, κΔθ)] # in place ifft
        # compute θ̇ in real space
        @. θ̇ = real(-u * ∂ˣθ + κΔθ + λ * (θ - θ^2/( c⁰* pⁱ )))
        # transitions
        for (j, θ2) in enumerate(θs)
            Qⁱʲ = Q[i, j]
            θ̇ .+= Qⁱʲ * θ2
        end
    end
    return nothing
end

# For conditional mean equations 
function rhs2!(θ̇s, θs, simulation_parameters)
    (; p, Q, us, c⁰, ∂ˣθ, c⁰, P, P⁻¹, ∂x, κ, Δ, κΔθ, λ) = simulation_parameters
    for (i, θ) in enumerate(θs)
        u = us[i]
        pⁱ = p[i]
        θ̇ = θ̇s[i]
        # dynamics
        P * θ # in place fft
        # ∇θ
        @. ∂ˣθ = ∂x * θ
        # κΔθ
        @. κΔθ = κ * Δ * θ
        # go back to real space 
        [P⁻¹ * field for field in (θ, ∂ˣθ, κΔθ)] # in place ifft
        # compute θ̇ in real space
        @. θ̇ = real(-u * ∂ˣθ + κΔθ - λ * (θ - pⁱ/( c⁰)))
        # transitions
        for (j, θ2) in enumerate(θs)
            Qⁱʲ = Q[i, j]
            θ̇ .+= Qⁱʲ * θ2
        end
    end
    return nothing
end

# For conditional mean equations 
function rhs3!(θ̇s, θs, simulation_parameters)
    (; p, Q, us, c⁰, ∂ˣθ, c⁰, P, P⁻¹, ∂x, κ, Δ, κΔθ, λ, φs) = simulation_parameters
    for (i, θ) in enumerate(θs)
        u = us[i]
        φ = φs[i]
        pⁱ = p[i]
        θ̇ = θ̇s[i]
        # dynamics
        P * θ # in place fft
        # ∇θ
        @. ∂ˣθ = ∂x * θ
        # κΔθ
        @. κΔθ = κ * Δ * θ
        # go back to real space 
        [P⁻¹ * field for field in (θ, ∂ˣθ, κΔθ)] # in place ifft
        # compute θ̇ in real space
        @. θ̇ = real(-u * ∂ˣθ + κΔθ + λ * (θ - θ^2/( c⁰* pⁱ ) * (θ * φ / (pⁱ)^2) ))
        # transitions
        for (j, θ2) in enumerate(θs)
            Qⁱʲ = Q[i, j]
            θ̇ .+= Qⁱʲ * θ2
        end
    end
    return nothing
end