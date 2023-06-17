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
options = (; titlesize=60, xlabel="x", ylabel="y", xlabelsize=60, ylabelsize=60, xticklabelsize=60, yticklabelsize=60)
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
Colorbar(fig[1:2, 4]; limits=colorrange, colormap=:balance, flipaxis=false, ticklabelsize=50)

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
save("data/fig4.eps", fig)

##
fig2 = Figure(resolution= (1200,1000))
field_tmp = sum(Θₘ, dims=3)[:, :, 1]

colorrange = (-4.0/3, 4.0/3)
contourlevels = range(-4.0/3, 4.0/3, 20)
for index_choice in 1:2
    ax1 = Axis(fig2[1, index_choice])
    field_cont = Θⁱ[:, :, index_choice]
    heatmap!(ax1, x[:], y[:], field_cont, colormap=:balance, interpolate=true, colorrange=colorrange)
    contour!(ax1, x[:], y[:], field_cont, color=:black, levels=contourlevels, linewidth=1.0)
    hidedecorations!(ax1)  # hides ticks, grid and lables
    hidespines!(ax1)  # hide the frame
end
ax1 = Axis(fig2[2, 1])
field_cont = Θⁱ[:, :, 3]
heatmap!(ax1, x[:], y[:], field_cont, colormap=:balance, interpolate=true, colorrange=colorrange)
contour!(ax1, x[:], y[:], field_cont, color=:black, levels=contourlevels, linewidth=1.0)
hidedecorations!(ax1)  # hides ticks, grid and lables
hidespines!(ax1)  # hide the frame
ax = Axis(fig2[2,2])
colorrange = (-4.0, 4.0)
contourlevels = range(-4.0, 4.0, 20)
heatmap!(ax, field_tmp, colormap=colormap, interpolate=true, colorrange=colorrange)
contour!(ax, field_tmp, color=:black, levels=contourlevels, linewidth=1.0)
hidedecorations!(ax)  # hides ticks, grid and lables
hidespines!(ax)  # hide the frame
display(fig2)

save(pwd() * "/data/graphical_abstract.png", fig2)
