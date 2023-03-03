using GLMakie, HDF5
##
global_filepath = "/Users/andresouza/Desktop/"
# fid = h5open(global_filepath * "random_phase_block.hdf5", "r")
fid = h5open(global_filepath * "random_phase_block_with_source.hdf5", "r")
KK = read(fid["effective_diffusivity_operator"])
θ̅_timeseries = read(fid["ensemble_average_field"])
diffusivity_timeseries = read(fid["diffusivity_field"])
nonlocal_timeseries = read(fid["nonlocal_field"])
close(fid)


##
fig = Figure(resolution=(1400, 1100))
ax11 = Axis(fig[1, 1]; title="ensemble average")
ax12 = Axis(fig[1, 2]; title="x=0 slice")
ax21 = Axis(fig[2, 1]; title="diffusion")
ax22 = Axis(fig[2, 2]; title="nonlocal space kernel")
iend = size(θ̅_timeseries, 3)
N = size(θ̅_timeseries, 1)
x_A = (0:N-1) / N * 4π .- 2π
t_slider = Slider(fig[3, 1:2], range=1:iend, startvalue=0)
tindex = t_slider.value
colormap = :bone_1
field = @lift(θ̅_timeseries[:, :, $tindex])
field_slice = @lift($field[:, floor(Int, N / 2)])
# field_slice = @lift(mean($field[:, :], dims=2)[:])
# colorrange=(0.0, 1.0), 

colorrange = @lift((0, maximum($field)))
field_diffusion = @lift(diffusivity_timeseries[:, :, $tindex])
field_diffusion_slice = @lift($field_diffusion[:, floor(Int, N / 2)])
# field_diffusion_slice = @lift(mean($field_diffusion[:, :], dims = 2)[:])

approximate_field = @lift(nonlocal_timeseries[:, :, $tindex])
approximate_field_slice = @lift($approximate_field[:, floor(Int, N / 2)])
# approximate_field_slice = @lift(mean($approximate_field[:, :], dims = 2)[:])

heatmap!(ax11, x_A, x_A, field, colormap=colormap, interpolate=true, colorrange=colorrange)
heatmap!(ax21, x_A, x_A, field_diffusion, colormap=colormap, interpolate=true, colorrange=colorrange)
heatmap!(ax22, x_A, x_A, approximate_field, colormap=colormap, interpolate=true, colorrange=colorrange)

le = lines!(ax12, x_A, field_slice, color=:black)
ld = lines!(ax12, x_A, field_diffusion_slice, color=:red)
lnd = lines!(ax12, x_A, approximate_field_slice, color=:blue)

axislegend(ax12, [le, ld, lnd], ["ensemble", "diffusion", "nonlocal space"], position=:rt)

display(fig)
##
framerate = 30
timestamps = 1:iend
GLMakie.record(fig, "time_animation_diffusivities_with_source.mp4", timestamps;
    framerate=framerate) do t
    tindex[] = t
    nothing
end;