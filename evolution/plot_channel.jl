@info "opening channel file"
hfile = h5open(pwd() * "/data/channel.hdf5", "r")
l2error = read(hfile["l2 error"])
sampling_error = read(hfile["sampling error"])
local_error = read(hfile["local error"])
Ms = read(hfile["Ms"])
nstate_ensemble_means = read(hfile["equations"])
empirical_ensemble_mean = read(hfile["empirical"])
x = read(hfile["x"])[:]
y = read(hfile["y"])[:]
s = read(hfile["source"])
nstate_ensemble_mean_local = read(hfile["local"])
close(hfile)

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
save("data/fig6.eps", error_fig)
## 
# Interpolate field
Mnew = 256
Ω = S¹(2π) × S¹(2)
grid = FourierGrid(Mnew, Ω)
x = reshape(grid.nodes[1], (Mnew, 1))  
y = reshape(grid.nodes[2], (1, Mnew))
interpolated_mean_field = zeros(ComplexF64, Mnew, Mnew)
Mold = size(empirical_ensemble_mean)[1]
Moldhalf = floor(Int, Mold / 2) + 1
indices_start = 1:Moldhalf
indices_end = Mnew-Moldhalf+3:Mnew
indices = [indices_start; indices_end]
view(interpolated_mean_field, indices, indices) .= fft(empirical_ensemble_mean)
field_tmp1 = copy(real.(ifft(interpolated_mean_field)) * Mnew^2 / Mold^2)
index_choice = 3 # length(Ms)
view(interpolated_mean_field, indices, indices) .= fft(nstate_ensemble_means[:, :, index_choice])
field_cont = copy(real.(ifft(interpolated_mean_field)) * Mnew^2 / Mold^2)
view(interpolated_mean_field, indices, indices) .= fft(nstate_ensemble_mean_local)
field_tmp2 = copy(real.(ifft(interpolated_mean_field)) * Mnew^2 / Mold^2)
view(interpolated_mean_field, indices, indices) .= fft(s)
scale = 0.1
field_tmp3 = copy(real.(ifft(interpolated_mean_field)) * Mnew^2 / Mold^2) * scale
##
N = length(x)
Nd2 = floor(Int, N / 2) + 1
fig = Figure(resolution=(2100, 1000))
titlelables1 = ["N = $(Ms[index_choice]) State Ensemble Mean"]
options = (; titlesize=30, xlabel="x", ylabel="y", xlabelsize=40, ylabelsize=40, xticklabelsize=30, yticklabelsize=30)
mth = maximum(nstate_ensemble_means)
colorrange = (-mth, mth)
ax = Axis(fig[1, 1]; title=titlelables1[1], options...)
# field_cont = nstate_ensemble_means[:, 1:Nd2, index_choice]
heatmap!(ax, x[:], y[1:Nd2], field_cont[:, 1:Nd2], colormap=:balance, interpolate=true, colorrange=colorrange)
contour!(ax, x[:], y[1:Nd2], field_cont[:, 1:Nd2], color=:black, levels=10, linewidth=1.0)
ax = Axis(fig[1, 2]; title="Continuous Empirical Ensemble Mean", options...)
# field_tmp1 = empirical_ensemble_mean[:, 1:Nd2]
heatmap!(ax, x[:], y[1:Nd2], field_tmp1[:, 1:Nd2], colormap=:balance, interpolate=true, colorrange=colorrange)
contour!(ax, x[:], y[1:Nd2], field_tmp1[:, 1:Nd2], color=:black, levels=10, linewidth=1.0)
ax = Axis(fig[2, 2]; title="Local Diffusivity Ensemble Mean", options...)
# field_tmp2 = nstate_ensemble_mean_local[:, 1:Nd2]
heatmap!(ax, x[:], y[1:Nd2], field_tmp2[:, 1:Nd2], colormap=:balance, interpolate=true, colorrange=colorrange)
contour!(ax, x[:], y[1:Nd2], field_tmp2[:, 1:Nd2], color=:black, levels=10, linewidth=1.0)
ax = Axis(fig[2, 1]; title="Source x $scale", options...)
# field_tmp3 = s[:, 1:Nd2] * 0.1
heatmap!(ax, x[:], y[1:Nd2], field_tmp3[:, 1:Nd2], colormap=:balance, interpolate=true, colorrange=colorrange)
contour!(ax, x[:], y[1:Nd2], field_tmp3[:, 1:Nd2], color=:black, levels=10, linewidth=1.0)

Colorbar(fig[1:2, 3]; limits=colorrange, colormap=:balance, flipaxis=false, ticklabelsize=30)
display(fig)
##
save("data/fig5.eps", fig)
@info "done creating plots"