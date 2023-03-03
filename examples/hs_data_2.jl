using HDF5, GLMakie
filepath = "/Users/andresouza/Desktop/" # Change Me
filename = filepath * "small_planet_hs_high_rez.h5"
fid = h5open(filename, "r")
u = read(fid["u"])
v = read(fid["v"])
T = read(fid["T"])
rho = read(fid["rho"])
close(fid)

##
θlist = range(0, 2π, length=720)
ϕlist = range(0, π, length=360)

x = [sin(ϕ) * cos(θ) for θ in θlist, ϕ in ϕlist]
y = [sin(ϕ) * sin(θ) for θ in θlist, ϕ in ϕlist]
z = [cos(ϕ) for θ in θlist, ϕ in ϕlist]
totes_sim = 3000
##
fig = Figure(resolution=(2000, 1000))
ax_T = LScene(fig[1, 1], show_axis=false)
ax_u = LScene(fig[1, 2], show_axis=false)
ax_v = LScene(fig[2, 1], show_axis=false)
ax_rho = LScene(fig[2, 2], show_axis=false)
time_slider = Slider(fig[3, 1:2], range=1:totes_sim, startvalue=1, horizontal=true)
time_index = time_slider.value
surface!(ax_T, x, y, z, color=@lift(T[:, :, $time_index]), colormap=:thermometer, colorrange=(270, 310), shading=false)
surface!(ax_u, x, y, z, color=@lift(u[:, :, $time_index]), colormap=:balance, colorrange=(-30, 30), shading=false)
surface!(ax_v, x, y, z, color=@lift(v[:, :, $time_index]), colormap=:balance, colorrange=(-30, 30), shading=false)
surface!(ax_rho, x, y, z, color=@lift(rho[:, :, $time_index]), colormap=:bone_1, colorrange=(1.1, 1.27), shading=false)

rotation = (π / 5, π / 6, 0)
rotation = (π / 16, π / 6, 0)
for ax in [ax_T, ax_u, ax_v, ax_rho]
    rotate_cam!(ax.scene, rotation)
end

display(fig)

##
fig = Figure(resolution=(2000, 1000))
ax_T = Axis(fig[1, 1], show_axis=false)
ax_u = Axis(fig[1, 2], show_axis=false)
ax_v = Axis(fig[2, 1], show_axis=false)
ax_rho = Axis(fig[2, 2], show_axis=false)
time_slider = Slider(fig[3, 1:2], range=1:totes_sim, startvalue=1, horizontal=true)
time_index = time_slider.value
heatmap!(ax_T, θlist, ϕlist, @lift(T[:, :, $time_index]), colormap=:thermometer, colorrange=(270, 310), shading=false, interpolate = true)
heatmap!(ax_u, θlist, ϕlist, @lift(u[:, :, $time_index]), colormap=:balance, colorrange=(-30, 30), shading=false, interpolate = true)
heatmap!(ax_v, θlist, ϕlist, @lift(v[:, :, $time_index]), colormap=:balance, colorrange=(-30, 30), shading=false, interpolate = true)
heatmap!(ax_rho, θlist, ϕlist, @lift(rho[:, :, $time_index]), colormap=:bone_1, colorrange=(1.1, 1.27), shading=false, interpolate = true)

display(fig)

##
framerate = 30
timestamps = 1:totes_sim

GLMakie.record(fig, filepath * "time_animation_hs_high_rez_heatmap.mp4", timestamps;
    framerate=framerate) do t
    time_index[] = t
    nothing
end

