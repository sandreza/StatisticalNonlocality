κxx = 0.0 .+ 1 * local_diffusivity_tensor[:, :, 1, 1]
κyy = 0.0 .+ 1 * local_diffusivity_tensor[:, :, 2, 2]
κyx = 0.0 .+ 1 * local_diffusivity_tensor[:, :, 2, 1]
κxy = 0.0 .+ 1 * local_diffusivity_tensor[:, :, 1, 2]

axx = @. π^2 / 4 * cos(π * y)^2
axy = @. -π / 4 * cos(π * y) * sin(π * y)
ayy = @. 1/ 4 * sin(π * y)^2
extrema(abs.(axx .- κxx))
extrema(abs.(axy .- κxy))
extrema(abs.(ayy .- κyy))
extrema(abs.(axy .+ κyx))
##

##
fig = Figure(resolution=(2814, 1192))
titlelables = ["κ₁₁", "κ₂₂", "κ₂₁", "κ₁₂"]
options = (; titlesize=30, xlabel="x", ylabel="y", xlabelsize=40, ylabelsize=40, xticklabelsize=30, yticklabelsize=30)
ax11 = Axis(fig[1, 1]; title=titlelables[1], options...)
lines!(ax11, κxx[1, :], colormap=:balance, interpolate=true, colorrange=(-1, 1))
ax12 = Axis(fig[2, 1]; title=titlelables[2], options...)
lines!(ax12, κyy[1, :], colormap=:balance, interpolate=true, colorrange=(-1, 1))
ax13 = Axis(fig[1, 2]; title=titlelables[3], options...)
lines!(ax13, κyx[1, :], colormap=:balance, interpolate=true, colorrange=(-1, 1))
ax14 = Axis(fig[2, 2]; title=titlelables[4], options...)
lines!(ax14, κxy[1, :], colormap=:balance, interpolate=true, colorrange=(-1, 1))
display(fig)

##
fig = Figure()
for i in 1:10
    ax = Axis(fig[i, 1])
    heatmap!(ax, real.(us[4*i]))
end
display(fig)