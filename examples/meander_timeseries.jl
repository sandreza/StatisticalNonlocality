using JLD2

global_path = "/Users/andresouza/Desktop/Data/markov_state_data/"
filename = "meander_surface_timeseries.jld2"
jlfile = jldopen(global_path * filename, "r")

dt_t = copy(jlfile["dt"]) # in days
us_t = []
vs_t = []
ws_t = []
bs_t = []
for i in ProgressBar(eachindex(keys(jlfile["u"])))
    push!(us_t, jlfile["u"][string(i)])
    push!(vs_t, jlfile["v"][string(i)])
    push!(ws_t, jlfile["w"][string(i)])
    push!(bs_t, jlfile["b"][string(i)])
end

close(jlfile)

state_timeseries = []
for j in eachindex(us_t)
    push!(state_timeseries, [us_t[j], vs_t[j], ws_t[j], bs_t[j]])
end

filename = "meander_markov_states.jld2"
jlfile = jldopen(global_path * filename, "r")
current_state = jlfile["current_state"]
dt = jlfile["dt"]
us_m = []
vs_m = []
ws_m = []
bs_m = []
for i in ProgressBar(eachindex(keys(jlfile["u"])))
    push!(us_m, jlfile["u"][string(i)])
    push!(vs_m, jlfile["v"][string(i)])
    push!(ws_m, jlfile["w"][string(i)])
    push!(bs_m, jlfile["b"][string(i)])
end

snapshots = []
for j in eachindex(us_m)
    push!(snapshots, [us_m[j], vs_m[j], ws_m[j], bs_m[j]])
end

close(jlfile)
Q = generator(current_state; dt=dt)
p = steady_state(Q)
perron_frobenius = exp(Q * dt)
##
# viz
tfig = Figure()
ax11 = Axis(tfig[1, 1])
ax12 = Axis(tfig[1, 2])
ax21 = Axis(tfig[2, 1])
ax22 = Axis(tfig[2, 2])

sl = Slider(tfig[3, 1:2], range=eachindex(us_t), startvalue=1)
tindex = sl.value
colorrange = (-0.5, 0.5)
# field = @lift(bs[$tindex][:, :, end] .- (mean(bs[$tindex][:, :, end])-mean(mb)) - mb)
field11 = @lift(us_t[$tindex][:, 1:100, end-1])
field12 = @lift(vs_t[$tindex][:, 1:100, end-1])
field21 = @lift(ws_t[$tindex][:, 1:100, end-1])
field22 = @lift(bs_t[$tindex][:, 1:100, end-1])
heatmap!(ax11, field11, colormap=:balance, colorrange=colorrange, interpolate=true)
heatmap!(ax12, field12, colormap=:balance, colorrange=colorrange, interpolate=true)
heatmap!(ax21, field21, colormap=:balance, colorrange=(-1e-3, 1e-3), interpolate=true)
heatmap!(ax22, field22, colormap=:thermometer, colorrange=(0, 0.015), interpolate=true)
display(tfig)

##
fig = Figure(resolution=(2 * 1500, 500))
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])
ax3 = Axis(fig[1, 3])
for ax in [ax1, ax2, ax3]
    hidedecorations!(ax)
    hidespines!(ax)
end

# sl = Slider(fig[2, 1:3], range=eachindex(us_t), startvalue=1)
tindex = Observable(1) # sl.value

field11 = @lift(us_t[$tindex][:, 10:90, end-1])
field12 = @lift(vs_t[$tindex][:, 10:90, end-1])
field13 = @lift(bs_t[$tindex][:, 10:90, end-1])

heatmap!(ax1, field11, colormap=:balance, colorrange=(-0.5, 0.5), interpolate=true)
heatmap!(ax2, field12, colormap=:balance, colorrange=(-0.5, 0.5), interpolate=true)
heatmap!(ax3, field13, colormap=:thermometer, colorrange=(0, 0.015), interpolate=true)

display(fig)

frames = eachindex(bs_t)

record(fig, "/Users/andresouza/Desktop/" * "meander_state_timeseries.mp4", frames; framerate=20) do frame
    tindex[] = frame
end

##
shifty = 50  # 80 # 50
shiftx = 20 # 120 # 20 # 120
#u[1][shiftx, shifty, end-2] 
index1 = 2
index2 = 4
reaction_coordinate(u) = mean(u[index1][1+shiftx:5+shiftx, 1+shifty:5+shifty, end] .* u[index2][1+shiftx:5+shiftx, 1+shifty:5+shifty, end], dims=(1, 2))[1] # - mean(u[4][1+shiftx:15+shiftx, 1+shifty:15+shifty, end-2], dims =(1,2))[1] .* mean(u[2][1+shiftx:15+shiftx, 1+shifty:15+shifty, end-2], dims=(1, 2))[1]  # u[2][120, 30, end-0] * u[4][120, 30, end-0] # mean(u[1][:, 30, end], dims=1)[1] # * u[4][140, 60, end-2] # * u[1][130, 60, end-1] # * u[4][140+0, 60+0, end]  # distance(u, snapshots[4])# real(iV[1, argmin([distance(u, s) for s in snapshots])])
markov = [reaction_coordinate(snapshot) for snapshot in snapshots]
rtimeseries = [reaction_coordinate(state) for state in state_timeseries]
xs_m, ys_m = histogram(markov, normalization=p, bins=9, custom_range=extrema(rtimeseries))
xs_t, ys_t = histogram(rtimeseries, bins=9, custom_range=extrema(rtimeseries))
fig = Figure(resolution=(1500, 1000))
kwargs = (; ylabel="probability", titlesize=30, ylabelsize=30, xticklabelsize=30, yticklabelsize=30)
ax1 = Axis(fig[1, 1]; title="Ensemble Statistics", kwargs...)
ax2 = Axis(fig[1, 2]; title="Temporal Statistics", kwargs...)
barplot!(ax1, xs_m, ys_m, color=:red)
barplot!(ax2, xs_t, ys_t, color=:blue)

for ax in [ax1, ax2]
    x_min = minimum([minimum(xs_m), minimum(xs_t)])
    x_max = maximum([maximum(xs_m), maximum(xs_t)])
    y_min = minimum([minimum(ys_m), minimum(ys_t)])
    y_max = maximum([maximum(ys_m), maximum(ys_t)])
    xlims!(ax, (x_min, x_max))
    ylims!(ax, (y_min, y_max))
end

display(fig)
ensemble_mean = sum(p .* markov)
temporal_mean = mean(rtimeseries)
ensemble_variance = sum(p .* (markov .^ 2)) - sum(p .* markov)^2
temporal_variance = mean(rtimeseries .^ 2) - mean(rtimeseries)^2
println("The ensemble mean is ", ensemble_mean)
println("The temporal mean is ", temporal_mean)
println("The null hypothesis is ", mean(markov))
println("The ensemble variance is ", ensemble_variance)
println("The temporal variance is ", temporal_variance)
println("The null hypothesis is ", var(markov))
println("The absolute error between the ensemble and temporal means is ", abs(ensemble_mean - temporal_mean))
println("The relative error between the ensemble and temporal mean are ", 100 * abs(ensemble_mean - temporal_mean) / abs(temporal_mean), " percent")
println("The relative error between the ensemble and temporal variances are ", 100 * abs(ensemble_variance - temporal_variance) / temporal_variance, " percent")

##
println("ensemble mean:",ensemble_mean)
println("temporal mean:",temporal_mean)
println("ensemble standard devation:", sqrt(ensemble_variance))
println("temporal standard devation:", sqrt(temporal_variance))

##
total = 60
auto_correlation_timeseries = zeros(total)
for s in 0:total-1
    auto_correlation_timeseries[s+1] = mean(rtimeseries[s+1:end] .* rtimeseries[1:end-s])
end
auto_correlation_timeseries .-= mean(rtimeseries)^2
auto_correlation_timeseries .*= 1 / auto_correlation_timeseries[1]

auto_correlation_snapshots = zeros(total)
val = [reaction_coordinate(snapshot) for snapshot in snapshots]
Pτ = perron_frobenius * 0 + I
perron_frobenius_scaled = perron_frobenius^(floor(Int, dt_t / dt))
for i in ProgressBar(0:total-1)
    auto_correlation_snapshots[i+1] = sum(val' * Pτ * (p .* val))
    Pτ *= perron_frobenius_scaled
end
auto_correlation_snapshots .= auto_correlation_snapshots .- sum(val .* p)^2;
auto_correlation_snapshots .*= 1.0 / auto_correlation_snapshots[1];
##
kwargs = (; titlesize=30, xlabelsize=30, ylabelsize=30, xticklabelsize=30, yticklabelsize=30)
auto_fig = Figure(resolution=(800, 800))
ax1 = Axis(auto_fig[1, 1]; title="Autocorrelation", xlabel="Days", ylabel="Dimensionless", kwargs...)
l1 = lines!(ax1, dt_t .* (0:total-1), auto_correlation_snapshots[:], color=:red)
l2 = lines!(ax1, dt_t .* (0:total-1), auto_correlation_timeseries[:], color=:blue)
Legend(auto_fig[1, 2], [l1, l2], ["Markov", "Timeseries"])
display(auto_fig)

##

# CHANGE ME HERE
kwargs = (; titlesize=30, xlabelsize=30, ylabelsize=30)
time_fig = Figure()
ax1 = Axis(time_fig[1, 2]; title="Autocorrelation", xlabel="Days", ylabel="Dimensionless", kwargs...)
ax2 = Axis(time_fig[1, 1]; title="Temporal Statistics", kwargs..., ylabel="Probability")
barplot!(ax2, xs_t, ys_t, color=:blue)
l2 = lines!(ax1, dt_t .* (0:total-1), auto_correlation_timeseries[:], color=:blue)
display(time_fig)

##
spatial_markov_correlation = Float64[]
spatial_timeseries_correlation = Float64[]
for i in 1:200
    reaction_coordinate1(u) = u[2][130, 60, end-0] # u[1][130, 60, end-0]
    reaction_coordinate2(u) = u[2][i, 60, end-0] # u[1][i, 60, end-0]
    reaction_coordinate(u) = reaction_coordinate1(u) * reaction_coordinate2(u) # distance(u, snapshots[4])# real(iV[1, argmin([distance(u, s) for s in snapshots])])
    markov = [reaction_coordinate(snapshot) for snapshot in snapshots]
    markov1 = [reaction_coordinate1(snapshot) for snapshot in snapshots]
    markov2 = [reaction_coordinate2(snapshot) for snapshot in snapshots]
    rtimeseries = [reaction_coordinate(state) for state in state_timeseries]
    rtimeseries1 = [reaction_coordinate1(state) for state in state_timeseries]
    rtimeseries2 = [reaction_coordinate2(state) for state in state_timeseries]
    ensemble_mean = (sum(p .* markov) - sum(p .* markov1) * sum(p .* markov2)) # / (sum(p .* markov1) * sum(p .* markov1))
    # ensemble_mean = (mean(markov) - mean(markov1) * mean(markov2)) 
    temporal_mean = (mean(rtimeseries) - mean(rtimeseries1) * mean(rtimeseries2)) # / (mean(rtimeseries1) * mean(rtimeseries2))
    push!(spatial_markov_correlation, ensemble_mean)
    push!(spatial_timeseries_correlation, temporal_mean)
end
##
longitude_c = range(0, 40, length=200)
latitude_c = range(-70, -50, length=100)

fig = Figure()
kwargs = (; titlesize=30, xlabelsize=30, ylabelsize=30)
ax1 = Axis(fig[1, 1]; title="Spatial Covariance", xlabel="Longitude [ᵒ]", ylabel="Covariance [m² s⁻²]", kwargs...)
l1 = lines!(ax1, longitude_c, spatial_markov_correlation, color=:red)
l2 = lines!(ax1, longitude_c, spatial_timeseries_correlation, color=:blue)
Legend(fig[1, 2], [l1, l2], ["Markov", "Timeseries"])
display(fig)
