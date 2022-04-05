# push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))
using JLD2
filepath = pwd()
# jlfile = jldopen("../data/nearly_local.jld2", "a+")
jlfile = jldopen("data/nonlocal_less_diffusivity.jld2", "a+")
# jlfile = jldopen("data/nonlocal.jld2", "a+")
# filepath = pwd()
# jlfile = jldopen("data/nearly_local.jld2", "a+")
ψ¹ = jlfile["streamfunction"]["ψ¹"]
ψ² = jlfile["streamfunction"]["ψ²"]
ψ³ = jlfile["streamfunction"]["ψ³"]
ψ⁴ = jlfile["streamfunction"]["ψ⁴"]
z = jlfile["grid"]["z"][:]
x = jlfile["grid"]["x"][:]

## Plot it
using GLMakie
fig = Figure(resolution = (1800, 1300), title = "Local Operators")
titlestring = "ψ¹"
ax1 = Axis(fig[1, 1], title = titlestring, titlesize = 30)
titlestring = "ψ²"
ax2 = Axis(fig[1, 3], title = titlestring, titlesize = 30)
titlestring = "ψ³"
ax3 = Axis(fig[2, 1], title = titlestring, titlesize = 30)
titlestring = "ψ⁴"
ax4 = Axis(fig[2, 3], title = titlestring, titlesize = 30)

colormap = :balance
hm1 = heatmap!(ax1, x, z, ψ¹, colormap = colormap, interpolate = true)

hm2 = heatmap!(ax2, x, z, ψ², colormap = colormap, interpolate = true)

hm3 = heatmap!(ax3, x, z, ψ³, colormap = colormap, interpolate = true)

hm4 = heatmap!(ax4, x, z, ψ⁴, colormap = colormap, interpolate = true)

Colorbar(fig[1, 2], hm1, height = Relative(3 / 4), width = 25, ticklabelsize = 30,
    labelsize = 30, ticksize = 25, tickalign = 1,)
Colorbar(fig[2, 2], hm2, height = Relative(3 / 4), width = 25, ticklabelsize = 30,
    labelsize = 30, ticksize = 25, tickalign = 1,)
Colorbar(fig[1, 4], hm3, height = Relative(3 / 4), width = 25, ticklabelsize = 30,
    labelsize = 30, ticksize = 25, tickalign = 1,)
Colorbar(fig[2, 4], hm4, height = Relative(3 / 4), width = 25, ticklabelsize = 30,
    labelsize = 30, ticksize = 25, tickalign = 1,)
display(fig)