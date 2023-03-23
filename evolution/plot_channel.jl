hfile = h5open("channel.hdf5")


using GLMakie
error_fig = Figure(resolution=(1612, 1180))
options = (; titlesize=30, xlabelsize=40, ylabelsize=40, xticklabelsize=40, yticklabelsize=40)
ax11 = Axis(error_fig[1, 1]; title="Relative L2 Error", xlabel="Number of States", ylabel="L2 Error (%)", options...)
scatter!(ax11, Ms, l2error; markersize=30, color=:black, label="N-State Model")
hlines!(ax11, [sampling_error], color=:red, linewidth=10, linestyle=:dash, label="Sampling Error")
hlines!(ax11, [local_error], color=:orange, linewidth=10, linestyle=:dash, label="Local Diffusivity Error")
ylims!(ax11, (0, 50))
ax11.xticks = (collect(Ms), string.(collect(Ms)))
axislegend(ax11, position=:rc, framecolor=(:grey, 0.5), patchsize=(40, 40), markersize=40, labelsize=50)
display(error_fig)
##
save("data/ensemble_mean_channel_error.png", error_fig)
##
Nd2 = floor(Int, N / 2) + 1
fig = Figure(resolution=(2100, 1000))
titlelables1 = ["N = $(Ms[end]) State Ensemble Mean"]
options = (; titlesize=30, xlabel="x", ylabel="y", xlabelsize=40, ylabelsize=40, xticklabelsize=30, yticklabelsize=30)
mth = maximum(nstate_ensemble_means)
colorrange = (-mth, mth)
ax = Axis(fig[1, 1]; title=titlelables1[1], options...)
index_choice = length(Ms)
field_cont = nstate_ensemble_means[:, 1:Nd2, index_choice]
heatmap!(ax, x[:], y[1:Nd2], field_cont, colormap=:balance, interpolate=true, colorrange=colorrange)
contour!(ax, x[:], y[1:Nd2], field_cont, color=:black, levels=10, linewidth=1.0)
ax = Axis(fig[1, 2]; title="Continuous Empirical Ensemble Mean", options...)
field_tmp = empirical_ensemble_mean[:, 1:Nd2]
heatmap!(ax, x[:], y[1:Nd2], field_tmp, colormap=:balance, interpolate=true, colorrange=colorrange)
contour!(ax, x[:], y[1:Nd2], field_tmp, color=:black, levels=10, linewidth=1.0)
ax = Axis(fig[2, 2]; title="Local Diffusivity Ensemble Mean", options...)
field_tmp = nstate_ensemble_mean_local[:, 1:Nd2]
heatmap!(ax, x[:], y[1:Nd2], field_tmp, colormap=:balance, interpolate=true, colorrange=colorrange)
contour!(ax, x[:], y[1:Nd2], field_tmp, color=:black, levels=10, linewidth=1.0)
#=
ax = Axis(fig[2, 1]; title="NState - Empirical", options...)
field_tmp = empirical_ensemble_mean[:, 1:Nd2] - nstate_ensemble_means[:, 1:Nd2, index_choice]
heatmap!(ax, x[:], y[1:Nd2], field_tmp, colormap=:balance, interpolate=true, colorrange = colorrange)
contour!(ax, x[:], y[1:Nd2], field_tmp, color=:black, levels=10, linewidth=1.0)
=#
ax = Axis(fig[2, 1]; title="Source x 0.1", options...)
field_tmp = s[:, 1:Nd2] * 0.1
heatmap!(ax, x[:], y[1:Nd2], field_tmp, colormap=:balance, interpolate=true, colorrange=colorrange)
contour!(ax, x[:], y[1:Nd2], field_tmp, color=:black, levels=10, linewidth=1.0)

Colorbar(fig[1:2, 3]; limits=colorrange, colormap=:balance, flipaxis=false, ticklabelsize=30)
display(fig)
##
save("data/ensemble_mean_channel_comparison.png", fig)