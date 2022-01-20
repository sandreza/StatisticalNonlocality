# push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))
using JLD2
filepath = pwd()
# jlfile = jldopen("data/nearly_local.jld2", "a+")
# jlfile = jldopen("data/nearly_local_symmetric.jld2", "a+")
jlfile = jldopen("data/nonlocal.jld2", "a+")
# jlfile = jldopen("../data/nonlocal_symmetric.jld2", "a+")
# jlfile = jldopen("data/nonlocal.jld2", "a+")
# filepath = pwd()
# jlfile = jldopen("data/nearly_local.jld2", "a+")
κ¹¹ = jlfile["localdiffusivity"]["κ11"]
κ¹² = jlfile["localdiffusivity"]["κ12"]
κ²¹ = jlfile["localdiffusivity"]["κ21"]
κ²² = jlfile["localdiffusivity"]["κ22"]
z = jlfile["grid"]["z"][:]
x = jlfile["grid"]["x"][:]
##
# Contruct local diffusivity estimate from flow field and γ
γ = jlfile["parameters"]["γ"]
ω = jlfile["parameters"]["ω"]
u¹ = jlfile["velocities"]["u¹"]
u² = jlfile["velocities"]["u²"]
v¹ = jlfile["velocities"]["v¹"]
v² = jlfile["velocities"]["v²"]

# integrating out the first dimension amounts to assuming that the dominant mode
# is the zero'th. 
N = length(x)
scale = (2 * (ω^2 + γ^2))
analytic_κ¹¹ = sum((γ * u¹ .* u¹ + γ * u² .* u²) ./ scale, dims = 1)[:] ./ N
analytic_κ¹² = sum((u¹ .* (γ * v¹ + ω * v²) + u² .* (γ * v² - ω * v¹)) ./ scale, dims = 1)[:] ./ N
analytic_κ²¹ = sum((v¹ .* (γ * u¹ + ω * u²) + v² .* (γ * u² - ω * u¹)) ./ scale, dims = 1)[:] ./ N
analytic_κ²² = sum((γ * v¹ .* v¹ + γ * v² .* v²) ./ scale, dims = 1)[:] ./ N

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

lines!(ax1, analytic_κ¹¹, z, color = :red)
scatter!(ax1, κ¹¹, z, color = :blue)

lines!(ax2, analytic_κ¹², z, color = :red)
scatter!(ax2, κ¹², z, color = :blue)

lines!(ax3, analytic_κ²¹, z, color = :red)
scatter!(ax3, κ²¹, z, color = :blue)

lines!(ax4, analytic_κ²², z, color = :red)
scatter!(ax4, κ²², z, color = :blue)

display(fig)