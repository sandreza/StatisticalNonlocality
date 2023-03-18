include("n_state.jl")
include("stochastic_advection.jl")

##
using GLMakie

keffs = Vector{Float64}[]
Ms = 0:7
for N in [1, 2, 3, 4]
    keff = n_state_keff(N; Ms = Ms)
    push!(keffs, keff)
end

Nlabels = ["N = 2", "N = 3", "N = 4", "N = 5"]
color_choices = [(:red, 0.5), (:blue, 0.5), (:green, 0.5), (:orange, 0.5)]
##
fig = Figure(resolution=(1600, 800))
options = (; title="Effective Diffusivities for Different Stochastic Models", titlesize=30, xlabel="Wavenumber", ylabel="Diffusivity", xlabelsize=40, ylabelsize=40, xticklabelsize=30, yticklabelsize=30)
ax = Axis(fig[1, 1]; options...)
scatter!(ax, [0], [1.0], label="Velocity Autocorrelation", color=:black, markersize=40, marker=:star5)
scatter!(ax, Ms[2:end], keff1, label = "Ensemble Average", color = :black, markersize = 30)
for i in eachindex(keffs)
    lines!(ax, Ms, keffs[i], label=Nlabels[i], color = color_choices[i], linewidth = 10)
    scatter!(ax, Ms, keffs[i], color=color_choices[i], markersize=20)
end
ax.xticks = (collect(Ms), string.(collect(Ms)))
axislegend(ax, position=:lb, framecolor=(:grey, 0.5), patchsize=(20, 20), markersize=30, labelsize=30)
display(fig)