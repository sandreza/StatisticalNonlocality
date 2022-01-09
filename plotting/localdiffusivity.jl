# push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))
using JLD2
filepath = pwd()
# jlfile = jldopen("../data/nearly_local.jld2", "a+")
jlfile = jldopen("data/nonlocal.jld2", "a+")
# filepath = pwd()
# jlfile = jldopen("data/nearly_local.jld2", "a+")
κ¹¹ = jlfile["localdiffusivity"]["κ11"]
κ¹² = jlfile["localdiffusivity"]["κ12"]
κ²¹ = jlfile["localdiffusivity"]["κ21"]
κ²² = jlfile["localdiffusivity"]["κ22"]
z = jlfile["grid"]["z"][:]
x = jlfile["grid"]["x"][:]
##
using GLMakie

fig = Figure(resolution = (1800, 1300), title = "Local Operators")
titlestring = "Kˣˣ"
ax1 = Axis(fig[1, 1], title = titlestring, titlesize = 30)
titlestring = "Kˣᶻ"
ax2 = Axis(fig[2, 1], title = titlestring, titlesize = 30)
titlestring = "Kᶻˣ"
ax3 = Axis(fig[1, 2], title = titlestring, titlesize = 30)
titlestring = "Kᶻᶻ"
ax4 = Axis(fig[2, 2], title = titlestring, titlesize = 30)

colormap = :thermal
colormap2 = :balance
A = maximum(κ¹¹)
k = 2π / (x[end] - x[1] + x[2]-x[1])
ℓ = π / (z[1] - z[end])
kdℓ = k/ℓ
f = maximum(κ²¹) / (A * kdℓ)

lines!(ax1, A * (sin.(ℓ * z) .^ 2), z, color = :red)
scatter!(ax1, κ¹¹, z, color = :blue)

lines!(ax2, -A * f * kdℓ * (sin.(ℓ * z) .* cos.(ℓ * z)) ./ maximum(sin.(ℓ * z) .* cos.(ℓ * z)), z, color = :red)
scatter!(ax2, κ¹², z, color = :blue)

lines!(ax3, A * f * kdℓ * (sin.(ℓ * z) .* cos.(ℓ * z)) ./ maximum(sin.(ℓ * z) .* cos.(ℓ * z)), z, color = :red)
scatter!(ax3, κ²¹, z, color = :blue)

lines!(ax4, A * (kdℓ)^2 * (cos.(ℓ * z) .^ 2), z, color = :red)
scatter!(ax4, κ²², z, color = :blue)

display(fig)