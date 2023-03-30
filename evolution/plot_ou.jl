@info "Plotting 3_state_ou.hdf5 data"
hfile = h5open(pwd() * "/data/3_state_ou.hdf5", "r")
x = read(hfile["x"])
y = read(hfile["y"])
Θⁱ = read(hfile["empirical"])
Θₘ = read(hfile["equations"])
stream_function = read(hfile["stream function"])
s = read(hfile["source"])
close(hfile)
##
fig = Figure(resolution=(2814, 1192))
titlelables = ["Θ₁", "Θ₂", "Θ₃"]
options = (; titlesize=30, xlabel="x", ylabel="y", xlabelsize=40, ylabelsize=40, xticklabelsize=30, yticklabelsize=30)
mth = 4.0 / 3
colorrange = (-mth, mth)
contourlevels = range(-mth, mth, 20)
for index_choice in 1:3
    ax1 = Axis(fig[1, index_choice]; title=titlelables[index_choice] * " : Empirical", options...)
    field_cont = Θⁱ[:, :, index_choice]
    heatmap!(ax1, x[:], y[:], field_cont, colormap=:balance, interpolate=true, colorrange=colorrange)
    contour!(ax1, x[:], y[:], field_cont, color=:black, levels=contourlevels, linewidth=1.0)
    ax2 = Axis(fig[2, index_choice]; title=titlelables[index_choice] * " : Equations", options...)
    field_tmp = Θₘ[:, :, index_choice]
    heatmap!(ax2, x[:], y[:], field_tmp, colormap=:balance, interpolate=true, colorrange=colorrange)
    contour!(ax2, x[:], y[:], field_tmp, color=:black, levels=contourlevels, linewidth=1.0)
end
Colorbar(fig[1:2, 4]; limits=colorrange, colormap=:balance, flipaxis=false, ticklabelsize=30)

index_choice = 5
colorrange = (-4.0, 4.0)
contourlevels = range(-4.0, 4.0, 20)
ax1 = Axis(fig[1, index_choice]; title="⟨θ⟩ : Empirical", options...)
field_cont = sum(Θⁱ, dims=3)[:, :, 1]
# :curl, :delta, :diverging_tritanopic_cwr_75_98_c20_n256, :diverging_tritanopic_cwr_75_98_c20_n256 
colormap = :balance
heatmap!(ax1, x[:], y[:], field_cont, colormap=colormap, interpolate=true, colorrange=colorrange)
contour!(ax1, x[:], y[:], field_cont, color=:black, levels=contourlevels, linewidth=1.0)
ax2 = Axis(fig[2, index_choice]; title="⟨θ⟩ : Equations", options...)
field_tmp = sum(Θₘ, dims=3)[:, :, 1]
heatmap!(ax2, x[:], y[:], field_tmp, colormap=colormap, interpolate=true, colorrange=colorrange)
contour!(ax2, x[:], y[:], field_tmp, color=:black, levels=contourlevels, linewidth=1.0)
Colorbar(fig[1:2, 6]; limits=colorrange, colormap=colormap, flipaxis=false, ticklabelsize=30)

index_choice = 7
colorrange = (-1,1)
colormap = :balance
ax1 = Axis(fig[1, index_choice]; title="Stream Function", options...)
heatmap!(ax1, x[:], y[:], stream_function, colormap=colormap, interpolate=true, colorrange=colorrange)
contour!(ax1, x[:], y[:], stream_function, color=:black, levels=10, linewidth=1.0)
ax2 = Axis(fig[2, index_choice]; title="Source", options...)
heatmap!(ax2, x[:], y[:], s, colormap=colormap, interpolate=true, colorrange=colorrange)
contour!(ax2, x[:], y[:], s, color=:black, levels=10, linewidth=1.0)
Colorbar(fig[1:2, 8]; limits=colorrange, colormap=colormap, flipaxis=false, ticklabelsize=30)


display(fig)
##
save(pwd() * "/data/3_state_ou.png", fig)
