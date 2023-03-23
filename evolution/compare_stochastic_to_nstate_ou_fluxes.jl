include("n_state_ou.jl")
include("stochastic_advection.jl")

##
using GLMakie
keffs = Vector{Float64}[]
Ms = 0:7
Ns = [1, 2, 3, 4, 14]
for N in Ns
    keff = n_state_keff(N; Ms = Ms)
    push!(keffs, keff)
end

Nlabels = ["N = $(Ns[1]+1)", "N = $(Ns[2]+1)", "N = $(Ns[3]+1)", "N = $(Ns[4]+1)", "N = $(Ns[end]+1)"]
color_choices = [(:red, 0.5), (:blue, 0.5), (:green, 0.5), (:orange, 0.5), (:black, 0.5)]
##
fig = Figure(resolution=(1718, 889))
options = (; title="Effective Diffusivities for Different Stochastic Models", titlesize=30, xlabel="Wavenumber", ylabel="Diffusivity", xlabelsize=40, ylabelsize=40, xticklabelsize=30, yticklabelsize=30)
ax = Axis(fig[1, 1]; options...)
scatter!(ax, [0], [1.0], label="Velocity Autocorrelation", color=:black, markersize=40, marker=:star5)
scatter!(ax, Ms[2:end], keff1, label = "Ensemble Average", color = :black, markersize = 40)
for i in eachindex(keffs)
    if i==5
        lines!(ax, Ms, keffs[i], label=Nlabels[i], color = color_choices[i], linewidth = 10, linestyle = :dot)
    else
        lines!(ax, Ms, keffs[i], label=Nlabels[i], color=color_choices[i], linewidth=10)
        scatter!(ax, Ms, keffs[i], color=color_choices[i], markersize=20)
    end
end
ax.xticks = (collect(Ms), string.(collect(Ms)))
axislegend(ax, position=:lb, framecolor=(:grey, 0.5), patchsize=(20, 20), markersize=20, labelsize=30)
display(fig)
(keffs[end][2:end] - keff1) ./ keff1 * 100
##
save("data/wavenumber_diffusivities.png", fig)
##
# kernels 
kernels = Vector{Float64}[]
keffs2 = Vector{Float64}[]
Ms = 0:10000
for N in Ns
    keff = n_state_keff(N; Ms=Ms)
    push!(keffs2, keff)
    tmp = [keff[1], keff[2:end]..., reverse(keff[2:end])...]
    kernel = circshift(real.(ifft(tmp)), Ms[end])
    push!(kernels, kernel)
end
##
Nlabels = ["N = $(Ns[1]+1)", "N = $(Ns[2]+1)", "N = $(Ns[3]+1)", "N = $(Ns[4]+1)", "N = $(Ns[end]+1)"]
color_choices = [(:red, 0.5), (:blue, 0.5), (:green, 0.5), (:orange, 0.5), (:black, 0.25)]

Ns = (length(kernels[1]))
Ω = S¹(2π)
grid = FourierGrid(Ns, Ω, arraytype=ArrayType)
nodes, wavenumbers = grid.nodes, grid.wavenumbers
Δx = nodes[1][2] - nodes[1][1]

fig = Figure(resolution=(1600, 800))
options = (; titlesize=30, xlabel="x-x'", ylabel="Amplitude", xlabelsize=40, ylabelsize=40, xticklabelsize=30, yticklabelsize=30)
for i in 1:4
    jj = (i - 1) % 2 + 1
    ii = (i - 1) ÷ 2 + 1
    ax = Axis(fig[ii, jj]; title="Kernel for " * Nlabels[i], options...)
    lines!(ax, nodes[1] .- π, kernels[i] / Δx , label=Nlabels[i], color = color_choices[i], linewidth = 10)
    lines!(ax, nodes[1] .- π, kernels[5] / Δx , label=Nlabels[5], color=color_choices[5], linewidth=10)
    axislegend(ax, position=:rt, framecolor=(:grey, 0.5), patchsize=(20, 20), markersize=30, labelsize=30)
end
display(fig)
##
save("data/kernels.png", fig)