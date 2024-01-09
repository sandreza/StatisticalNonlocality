using GLMakie, LinearAlgebra, StatisticalNonlocality, ProgressBars

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
κ = 1e-2
U = 0.01
kappas = n_state_keff(N; Ms=Ms, κ=κ, λ=0.0, γ=1.0, ϵ=√2, U=U)

function model(a,b)
    return @. U^2 / sqrt(1 +  a * Ms^2 + b * Ms^4)
end
function loss_function(a, b, kappas)
    tmp = model(a,b)
    norm(tmp ./ kappas .- 1)
end
# b = 0.0001
tmp = model(U^2 * 2,  κ^2)
fig = Figure() 
ax = Axis(fig[1,1]; title = "$N state ")
lines!(ax, log10.(Ms), log10.(kappas), color=(:red, 0.5), linewidth=5, label="N = $N")
ax = Axis(fig[1,2]; title = "model ")
lines!(ax, log10.(Ms), log10.(tmp), color=(:blue, 0.5), linewidth=5, label="N = $N")
ax = Axis(fig[2, 1]; title  = "together")
lines!(ax, log10.(Ms), log10.(kappas), color=(:red, 0.5), linewidth=5, label="N = $N")
lines!(ax, log10.(Ms), log10.(tmp), color=(:blue, 0.5), linewidth=5, label="N = $N")
ax = Axis(fig[2, 2]; title = "difference")
lines!(ax, log10.(Ms), log10.(kappas) - log10.(tmp), color=(:red, 0.5), linewidth=5, label="N = $N")
display(fig)
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