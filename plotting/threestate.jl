using JLD2
using GLMakie

filename = "three_state.jld2"
file = jldopen("../data/" * filename, "a+")

fluxkernel = file["kernel"]
λ = file["eigenvalues"]
σ = file["singularvalues"]

localλ = sum(fluxkernel, dims = 2)[1:2:end]
pλ = real.(reverse(λ[1:2:end])) # for plotting just need every other eigenvalue due to sin cos sym


options = (; xlabel = "Eigenvalue Index",
    xlabelcolor = :black, ylabel = "Eigenvalue Magnitude",
    ylabelcolor = :black, xlabelsize = 40, ylabelsize = 40,
    xticklabelsize = 25, yticklabelsize = 25,
    xtickcolor = :black, ytickcolor = :black,
    xticklabelcolor = :black, yticklabelcolor = :black,
    titlesize = 50)

fig = Figure()
ax1 = Axis(fig[1, 1]; options...)
sc1 = scatter!(ax1, pλ, color = :blue, markersize = 10)
sc2 = scatter!(ax1, localλ, color = :red, markersize = 10)
axislegend(ax1, [sc1, sc2], ["nonlocal", "local"], position = :rc)