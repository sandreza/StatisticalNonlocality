# push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))
using JLD2

# jlfile = jldopen("data/local.jld2", "a+")
jlfile = jldopen("data/nonlocal.jld2", "a+")
# jlfile = jldopen("data/nonlocal_symmetric.jld2", "a+")
# jlfile = jldopen("data/nonlocal_more_velocity.jld2")
EF¹¹ = jlfile["diffusivity"]["K11"]
EF¹² = jlfile["diffusivity"]["K12"]
EF²¹ = jlfile["diffusivity"]["K21"]
EF²² = jlfile["diffusivity"]["K22"]
x = jlfile["grid"]["x"]
z = jlfile["grid"]["z"]
N = length(x)
M = length(z)
tmpE = copy(EF¹¹)
EF¹¹ = reshape(permutedims(reshape(EF¹¹, (N, M, N, M)), (2, 1, 4, 3)), (N * M, N * M))
EF¹² = reshape(permutedims(reshape(EF¹², (N, M, N, M)), (2, 1, 4, 3)), (N * M, N * M))
EF²¹ = reshape(permutedims(reshape(EF²¹, (N, M, N, M)), (2, 1, 4, 3)), (N * M, N * M))
EF²² = reshape(permutedims(reshape(EF²², (N, M, N, M)), (2, 1, 4, 3)), (N * M, N * M))
# 2 goes to 1, 1 goes to 2, 4 goes to 3, 3 goes to 4
##

using GLMakie
fig = Figure(resolution = (1800, 1300), title = "Nonlocal Operators")
titlestring = "Kˣˣ"
ax1 = Axis(fig[1, 1], title = titlestring, titlesize = 30)
titlestring = "Kˣᶻ"
ax2 = Axis(fig[1, 3], title = titlestring, titlesize = 30)
titlestring = "Kᶻˣ"
ax3 = Axis(fig[2, 1], title = titlestring, titlesize = 30)
titlestring = "Kᶻᶻ"
ax4 = Axis(fig[2, 3], title = titlestring, titlesize = 30)

colormap = :thermal
colormap2 = :balance
hm1 = heatmap!(ax1, EF¹¹, colormap = colormap)
ax1.yreversed = true

hm2 = heatmap!(ax2, EF¹², colormap = colormap2)
ax2.yreversed = true

hm3 = heatmap!(ax3, EF²¹, colormap = colormap2)
ax3.yreversed = true

hm4 = heatmap!(ax4, EF²², colormap = colormap)
ax4.yreversed = true

Colorbar(fig[1, 2], hm1, height = Relative(3 / 4), width = 25, ticklabelsize = 30,
    labelsize = 30, ticksize = 25, tickalign = 1,)
Colorbar(fig[2, 2], hm2, height = Relative(3 / 4), width = 25, ticklabelsize = 30,
    labelsize = 30, ticksize = 25, tickalign = 1,)
Colorbar(fig[1, 4], hm3, height = Relative(3 / 4), width = 25, ticklabelsize = 30,
    labelsize = 30, ticksize = 25, tickalign = 1,)
Colorbar(fig[2, 4], hm4, height = Relative(3 / 4), width = 25, ticklabelsize = 30,
    labelsize = 30, ticksize = 25, tickalign = 1,)
display(fig)

##
using CairoMakie
fig = Figure(resolution = (1800, 1300), title = "Nonlocal Operators")
titlestring = "Kᶻᶻ"
ax4 = Axis(fig[1,1], title = titlestring, titlesize = 30)
sizeval = 30
colormap = :thermal
colormap2 = :balance
options = (; xlabelsize = sizeval, ylabelsize = sizeval, xticklabelsize = sizeval, yticklabelsize = sizeval)

hm4 = heatmap!(ax4, EF²²; colormap = colormap, options...)
ax4.yreversed = true

Colorbar(fig[1, 2], hm4, height = Relative(3 / 4), width = 25, ticklabelsize = 30,
    labelsize = 30, ticksize = 25, tickalign = 1,)
display(fig)
save("data/fig2.eps", fig)