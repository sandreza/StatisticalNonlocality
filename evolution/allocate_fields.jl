function allocate_2D_fields(N, M; arraytype = Array)
    # velocity
    ψ = arraytype(zeros(ComplexF64, N, N))
    ψ2 = arraytype(zeros(ComplexF64, N, N))
    # theta
    θs = [similar(ψ) for i in 1:M]
    θv = similar(ψ)
    ∂ˣθ = similar(ψ)
    ∂ʸθ = similar(ψ)
    κΔθ = similar(ψ)
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
    us = [arraytype(zeros(ComplexF64, N, N)) for i in 1:M]
    vs = [arraytype(zeros(ComplexF64, N, N)) for i in 1:M]
    return (; ψ, ψ2, θs, θv, ∂ˣθ, ∂ʸθ, κΔθ, θ̇s, s, θ̅, k₁, k₂, k₃, k₄, θ̃, uθ, vθ, ∂ˣuθ, ∂ʸvθ, us, vs)
end