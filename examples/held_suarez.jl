using HDF5
filename = "even_high_rez_hs.h5"
fid = h5open(filename, "r")
θlist = read(fid["grid"]["θlist"])
ϕlist = read(fid["grid"]["ϕlist"])
rlist = read(fid["grid"]["rlist"])
tic = time()
T_timeseries = []
u_timeseries = []
v_timeseries = []
rho_timeseries = []
tic = time()
for i in 1:length(fid["T"])
    push!(T_timeseries, read(fid["T"][string(i)]))
    push!(u_timeseries, read(fid["u"][string(i)]))
    push!(v_timeseries, read(fid["v"][string(i)]))
    push!(rho_timeseries, read(fid["rho"][string(i)]))

    toc = time()
    if toc - tic > 1
        println("currently at timestep ", i, " out of ", length(fid["T"]))
        tic = toc
    end
end

##
x = [sin(ϕ) * cos(θ) for θ in θlist, ϕ in ϕlist]
y = [sin(ϕ) * sin(θ) for θ in θlist, ϕ in ϕlist]
z = [cos(ϕ) for θ in θlist, ϕ in ϕlist]

fig = Figure()
ax_T = LScene(fig[1, 1], show_axis=false)
ax_u = LScene(fig[1, 2], show_axis=false)
ax_v = LScene(fig[2, 1], show_axis=false)
ax_rho = LScene(fig[2, 2], show_axis=false)
time_slider = Slider(fig[3, 1:2], range=1:length(T_timeseries), startvalue=1, horizontal=true)
time_index = time_slider.value
surface!(ax_T, x, y, z, color=@lift(T_timeseries[$time_index]), colormap=:afmhot, colorrange=(270, 304), shading=false)
surface!(ax_u, x, y, z, color=@lift(u_timeseries[$time_index]), colormap=:balance, colorrange=(-30, 30), shading=false)
surface!(ax_v, x, y, z, color=@lift(v_timeseries[$time_index]), colormap=:balance, colorrange=(-30, 30), shading=false)
surface!(ax_rho, x, y, z, color=@lift(rho_timeseries[$time_index]), colormap=:bone_1, colorrange=(1.06, 1.2), shading=false)

##
framerate = 3 * 30
timestamps = 1:length(T_timeseries)

record(fig, "time_animation_hs.mp4", timestamps;
    framerate=framerate) do t
    time_index[] = t
end

##
fig = Figure()
ax_u = LScene(fig[1, 1], show_axis=false)
ax_state = LScene(fig[1, 2], show_axis=false)
time_slider = Slider(fig[2, 1:2], range=1:length(T_timeseries), startvalue=1, horizontal=true)
time_index = time_slider.value
surface!(ax_u, x, y, z, color=@lift(u_timeseries[$time_index]), colormap=:balance, colorrange=(-30, 30), shading=false)
surface!(ax_state, x, y, z, color=@lift(snapshots[cstate[$time_index]]), colormap=:balance, colorrange=(-30, 30), shading=false)

framerate = 3 * 30
timestamps = 1:length(T_timeseries)

record(fig, "time_animation_hs_markov.mp4", timestamps;
    framerate=framerate) do t
    time_index[] = t
end

##
fig = Figure()
ax_11 = LScene(fig[1, 1], show_axis=false)
ax_12 = LScene(fig[1, 2], show_axis=false)
ax_21 = LScene(fig[2, 1], show_axis=false)
ax_22 = LScene(fig[2, 2], show_axis=false)
surface!(ax_11, x, y, z, color=snapshots[1], colormap=:balance, colorrange=(-30, 30), shading=false)
surface!(ax_12, x, y, z, color=snapshots[2], colormap=:balance, colorrange=(-30, 30), shading=false)
surface!(ax_21, x, y, z, color=snapshots[3], colormap=:balance, colorrange=(-30, 30), shading=false)
surface!(ax_22, x, y, z, color=snapshots[4], colormap=:balance, colorrange=(-30, 30), shading=false)
