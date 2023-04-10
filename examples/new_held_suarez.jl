using HDF5, GLMakie, LinearAlgebra, ProgressBars, MarkovChainHammer
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
display(fig)
##
snapshots = [T_timeseries[i] for i in [2000, 4000, 6000, 8000]]
markov_embedding = zeros(Int, length(T_timeseries))
for i in ProgressBar(eachindex(markov_embedding))
    markov_embedding[i] = argmin([norm(snapshot .- T_timeseries[i]) for snapshot in snapshots])
end
##
using MarkovChainHammer.TransitionMatrix: perron_frobenius, generator
perron_frobenius(markov_embedding)
Q = generator(markov_embedding; dt = 0.003) 

##
kwargs = (; title = "Embedding", titlesize=40, ylabelsize=40, xgridstyle=:dash, ygridstyle=:dash, xtickalign=1,
    xticksize=20, ytickalign=1, yticksize=20, xticklabelsize=40, yticklabelsize=40)
fig = Figure(resolution=(2400, 1070))
ax_u = LScene(fig[1, 1], show_axis=false)
ax_state = LScene(fig[1, 2], show_axis=false)
ax_embedding = Axis(fig[1, 3]; kwargs...)
last_ind = 2000-1
time_slider = Slider(fig[2, 1:3], range=1:last_ind, startvalue=1, horizontal=true)
time_index = time_slider.value
colorrange = extrema(T_timeseries[1])
surface!(ax_u, x, y, z, color=@lift(T_timeseries[$time_index]), colormap=:afmhot, colorrange=colorrange, shading=false)
surface!(ax_state, x, y, z, color=@lift(snapshots[markov_embedding[$time_index]]), colormap=:afmhot, colorrange=colorrange, shading=false)
xlims!(ax_embedding, (1, last_ind))
ylims!(ax_embedding, (0.5, 4.5))
ax_embedding.yticks = ([1, 2, 3, 4], ["1", "2", "3", "4"])
ys = @lift(markov_embedding[1:$time_index])
scatter!(ax_embedding, ys, color=:black, markersize=20)


display(fig)
framerate = 3 * 30
timestamps = 1:last_ind
##
record(fig, "time_animation_hs_markov_brown.mp4", timestamps;
    framerate=framerate) do t
    time_index[] = t
end

##
fig = Figure(resolution=(2802, 886))
for i in 1:4
    ax = LScene(fig[1, i]; show_axis=false)
    surface!(ax, x, y, z, color=snapshots[i], colormap=:afmhot, colorrange=colorrange, shading=false)
end
display(fig)