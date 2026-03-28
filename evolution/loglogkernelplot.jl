using GLMakie, LinearAlgebra, StatisticalNonlocality, ProgressBars

function ou_transition_matrix(n)
    Mⱼₖ = zeros(n + 1, n + 1)
    δ(j, k) = (j == k) ? 1 : 0

    for j in 0:n, k in 0:n
        jj = j + 1
        kk = k + 1
        Mⱼₖ[jj, kk] =
            (-n * δ(j, k) + k * δ(j + 1, k) + (n - k) * δ(j - 1, k)) / 2
    end
    return Mⱼₖ
end

function n_state_keff(N; Ms = 1:7, κ = 0.01, λ = 0.0, γ = 1.0, ϵ = √2, U = 1.0)
    Δx = 2 / √N
    uₘ = 1 / sqrt(γ * 2 / ϵ^2) * [Δx * (i - N / 2) for i in 0:N] * U
    Q = ou_transition_matrix(N) .* γ
    Λ, V = eigen(Q)
    V⁻¹ = inv(V)
    U = V * Diagonal(uₘ) * V⁻¹
    vtop = U[end, 1:end-1]
    vbottom = U[1:end-1, end]
    keff = Float64[]
    for k in ProgressBar(Ms)
        vbot = im * k * U[1:end-1, 1:end-1] + Diagonal(Λ[1:end-1] .- λ .- κ * k^2)
        𝒦ₘ = -real(vtop' * (vbot \ vbottom))
        push!(keff, 𝒦ₘ)
    end
    return keff
end

Ms = range(1e-4, 1e-2, length = 10)
Ms = [Ms..., range(0.01, 1, length = 2000)...]
Ms = [Ms..., range(1, 10000, length= 2000)...]

U = sqrt(3)
fig = Figure(resolution = (2000, 2000)) 
keffs = Vector{Float64}[]
for i in 1:9
    ii = (i - 1) ÷ 3 + 1
    jj = (i - 1) % 3 + 1

    ax = Axis(fig[ii, jj]; xlabel = "log10(k)", ylabel = "log10(κ)")
    N = 4 * i
    kappas = n_state_keff(N; Ms=Ms, κ=0.01, λ=0.0, γ=1.0, ϵ=√2, U=U)
    push!(keffs, kappas)
    lines!(ax, log10.(Ms), log10.(kappas), color = (:blue, 0.5), linewidth = 5, label = "N = $N")
    # xlims!(ax, (-4, 4))
    # ylims!(ax, (-4, 4))
    xlims!(ax, (-2, 4))
    ylims!(ax, (-5, ceil(Int, log10(U^2)+1)))
    N = 4 * i + 1
    kappas = n_state_keff(N; Ms=Ms, κ=0.01, λ=0.0, γ=1.0, ϵ=√2, U=U)
    push!(keffs, kappas)
    lines!(ax, log10.(Ms), log10.(kappas), color=(:red, 0.5), linewidth=5, label="N = $N")
    axislegend(ax, position=:lb, framecolor=(:grey, 0.5), patchsize=(20, 20), markersize=20, labelsize=30)
    xlims!(ax, (-2, 4))
    ylims!(ax, (-5, ceil(Int, log10(U^2)+1)))
end
display(fig)
##
save("data/loglog_kernel.png", fig)

##
N = 100
κ = 1e-3
U = 2.0
kappasN = n_state_keff(N; Ms=Ms, κ=κ, λ=0.0, γ=1.0, ϵ=√2, U=U)
kappasNp1 = n_state_keff(N+1; Ms=Ms, κ=κ, λ=0.0, γ=1.0, ϵ=√2, U=U)
kappas = (kappasN + kappasNp1)/2

kappas50 = n_state_keff(50; Ms=Ms, κ=κ, λ=0.0, γ=1.0, ϵ=√2, U=U)

function model(a,b)
    return @. U^2 / sqrt(1 +  a * Ms^2 + b * Ms^4)
end

function model2(a, b)
    return @. 2 * U^2 / (κ * Ms^2 + sqrt((2 + κ * Ms^2)^2 + 4 * (U * Ms)^2))
end

function model3(a, b)
    numerator = @. U^2 * sqrt((κ * Ms^2 + 3)^2 + 4 * (U * Ms)^2 + κ * Ms^2 + 1)
    denominator = @. (κ * Ms^2 + 1) * sqrt((κ * Ms^2 + 3)^2 + 4 * (U * Ms)^2 + (κ * Ms^2 + 1)^2 + 2 * (U * Ms)^2)
    return numerator ./ denominator
end

function model4(a, b)
    return @. 2 * U^2 / ((κ * Ms^2 + 1) + sqrt((κ * Ms^2 + 1)^2 + 8 * (U * Ms)^2))
end

function D(n)
    return @. ((κ * Ms^2 + n - 1) + sqrt((κ * Ms^2 + n - 1)^2 + 4 * (U * Ms)^2 * n)) / 2
end

function DD(n)
    return @. ((κ * Ms^2 + n ) + sqrt((κ * Ms^2 + n)^2 + 4 * (U * Ms)^2 * n)) / 2
end

function model5(a, b)
    D2 = D(3)
    return @. U^2 / (κ * Ms^2 + 1 + (U * Ms)^2 * 2 / D2)
end

function loss_function(a, b, kappas)
    tmp = model(a,b)
    norm(tmp ./ kappas .- 1)
end
# b = 0.0001
tmp = model5(U^2 * 2,  κ^2) # (model5(U^2 * 2,  κ^2) + model4(U^2 * 2,  κ^2)) / 2
fig = Figure() 
ax = Axis(fig[1,1]; title = "$N state ", xlabel = "log₁₀(k)", ylabel = "log₁₀(κ_E)")
lines!(ax, log10.(Ms), log10.(kappasN), color=(:red, 0.5), linewidth=5, label="N = $N")
lines!(ax, log10.(Ms), log10.(kappasNp1), color=(:blue, 0.5), linewidth=5, label="N = $(N+1)")
lines!(ax, log10.(Ms), log10.(kappas), color=(:purple, 0.5), linewidth=5, label="N = ($N + $(N+1))/2")
lines!(ax, log10.(Ms), log10.(kappas50), color=(:black, 0.5), linewidth=5, label="N = 50")
axislegend(ax, position=:lb)
ax = Axis(fig[1,2]; title = "model", xlabel = "log₁₀(k)", ylabel = "log₁₀(κ_E)")
lines!(ax, log10.(Ms), log10.(tmp), color=(:blue, 0.5), linewidth=5, label="model")
axislegend(ax, position=:lb)
ax = Axis(fig[2, 1]; title  = "together", xlabel = "log₁₀(k)", ylabel = "log₁₀(κ_E)")
lines!(ax, log10.(Ms), log10.(kappas), color=(:red, 0.5), linewidth=5, label="data (N = ($N + $(N+1))/2)")
lines!(ax, log10.(Ms), log10.(tmp), color=(:blue, 0.5), linewidth=5, label="model")
axislegend(ax, position=:lb)
ax = Axis(fig[2, 2]; title = "difference", xlabel = "log₁₀(k)", ylabel = "log₁₀(κ_data) - log₁₀(κ_model)")
lines!(ax, log10.(Ms), log10.(kappas) - log10.(tmp), color=(:red, 0.5), linewidth=5, label="difference")
axislegend(ax, position=:rt)
display(fig)
save("data/loglog_kernel_model.png", fig)

##
as = reshape([10^i for i in -5:0.1:5], :, 1)
bs = reshape([10^i for i in -5:0.1:5], 1, :)
bestinds = argmin(loss_function.(as, bs, Ref(kappas)))
a = as[bestinds[1]]
b = bs[bestinds[2]]
##
tmp = model(a, b)
lines(log10.(Ms), log10.(kappas), color=(:red, 0.5), linewidth=5, label="N = $N")
lines!(log10.(Ms), log10.(tmp), color=(:blue, 0.5), linewidth=5, label="N = $N")