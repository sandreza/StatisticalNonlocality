

function periodic_drift(c, ϵ, Δt, M, iend)
    process = zeros(Float64, M, iend)
    for j in 1:M
        process[j, 1] = rand(Uniform(0, 2π))
        for i in 2:iend
            process[j, i] = (process[j, i-1] + c * Δt + ϵ * randn() * sqrt(2Δt)) % 2π
        end
    end
    return process
end

function update_channel_flow_field!(us, vs, ψ, process_n, x, y, kˣ, kʸ, ∂y, ∂x, P, P⁻¹, U)
    for i in eachindex(process_n)
        a = process_n[i]
        @. ψ = U * cos(kˣ[2] * x + a) * sin(kʸ[2] * y)
        P * ψ  # in place fft
        @. us[i] = -1.0 * (∂y * ψ)
        @. vs[i] = (∂x * ψ)
        P⁻¹ * us[i]
        P⁻¹ * vs[i]
    end
end

function continuous_channel(N, M; U=1.0, ϵ=π / sqrt(8), c=-π / 2, source_index=3, arraytype=Array)
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

    ## Allocate Arrays
    (; ψ, ψ2, θs, θv, ∂ˣθ, ∂ʸθ, κΔθ, θ̇s, s, θ̅, k₁, k₂, k₃, k₄, θ̃, uθ, vθ, ∂ˣuθ, ∂ʸvθ, us, vs) = allocate_2D_fields(N, M; arraytype=arraytype)

    ## plan ffts
    P = plan_fft!(ψ)
    P⁻¹ = plan_ifft!(ψ)

    ## Allocate Fields with specific values
    [us[i] .= -kʸ[2] * U * cos.(kˣ[2] * x .+ rand(Uniform(0, 2π))) .* cos.(kʸ[2] .* y) for i in 1:M]
    [vs[i] .= -kˣ[2] * U * sin.(kˣ[2] * x .+ rand(Uniform(0, 2π))) .* sin.(kʸ[2] .* y) for i in 1:M]
    @. s = sin(kˣ[2] * x) * sin(kʸ[source_index] * y) + cos(kˣ[3] * x) * sin(kʸ[source_index] * y)

    ## Set Initial Theta Equal to Diffusive Solution
    tmp = (kˣ[source_index]^2 + kʸ[source_index]^2)
    for (i, θ) in enumerate(θs)
        θ .= (s ./ (tmp * κ))
    end

    ## Timestepping 
    Δx = x[2] - x[1]
    cfl = 1.0
    advective_u_Δt = cfl * Δx / maximum([maximum(real.(abs.(us[i]))) for i in 1:M])
    advective_v_Δt = cfl * Δx / maximum([maximum(real.(abs.(vs[i]))) for i in 1:M])
    diffusive_Δt = cfl * Δx^2 / κ
    Δt = min(advective_u_Δt, advective_v_Δt, diffusive_Δt)

    return (; ψ, x, y, kˣ, kʸ, θs, us, vs, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ, θ̇s, θ̅, k₁, k₂, k₃, k₄, θ̃, Δt)
end


function ou_process(γ, ϵ, Δt, M, iend)
    process = zeros(Float64, M, iend)
    for j in 1:M
        process[j, 1] = randn()
        for i in 2:iend
            process[j, i] = process[j, i-1] * (1-γ * Δt) + ϵ * randn() * sqrt(Δt)
        end
    end
    return process
end

function nstate_ou_process(Q, uₘ, Δt, M, iend)
    process = zeros(Float64, M, iend)
    process_index = zeros(Int64, M, iend)
    for j in 1:M
        markov_chain = generate(exp(Q * Δt), iend)
        process_index[j, :] .= markov_chain
        for i in 1:iend
            process[j, i] = uₘ[markov_chain[i]]
        end
    end
    return process, process_index
end

function update_ou_flow_field!(us, vs, ψ, process_n, x, y, kˣ, kʸ, ∂y, ∂x, P, P⁻¹)
    for i in eachindex(process_n)
        a = process_n[i]
        @. ψ = a * cos(kˣ[2] * x) * cos(kʸ[2] * y)
        P * ψ  # in place fft
        @. us[i] = -1.0 * (∂y * ψ)
        @. vs[i] = (∂x * ψ)
        P⁻¹ * us[i]
        P⁻¹ * vs[i]
    end
end