

##
snapshots = markov_states
state_timeseries = state
reaction_coordinate(u) = (u[1] <= 0) # real(iV[1, argmin([norm(u - s) for s in markov_states])]) # u[3] # 
markov = [reaction_coordinate(snapshot) for snapshot in snapshots]
rtimeseries = [reaction_coordinate(state) for state in state_timeseries]
xs_m, ys_m = histogram(markov, normalization=p, bins=20, custom_range=extrema(rtimeseries))
xs_t, ys_t = histogram(rtimeseries, bins=20, custom_range=extrema(rtimeseries))
fig = Figure()
kwargs = (; ylabel="probability", titlesize=30, ylabelsize=30)
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
println("The relative error between the ensemble and temporal variances are ", 100 * abs(ensemble_variance - temporal_variance) / temporal_variance, " percent")

##
markov = [reaction_coordinate(snapshot) for snapshot in snapshots]
rtimeseries = [reaction_coordinate(state) for state in state_timeseries]

total = 800# *3*3
auto_correlation_timeseries = zeros(total)
for s in ProgressBar(0:total-1)
    auto_correlation_timeseries[s+1] = mean(rtimeseries[s+1:end] .* rtimeseries[1:end-s])
end
auto_correlation_timeseries .-= mean(rtimeseries)^2
auto_correlation_timeseries .*= 1 / auto_correlation_timeseries[1]

auto_correlation_snapshots = zeros(total)
val = [reaction_coordinate(snapshot) for snapshot in snapshots]

Pτ = perron_frobenius * 0 + I
for i in ProgressBar(0:total-1)
    auto_correlation_snapshots[i+1] = sum(val' * Pτ * (p .* val))
    Pτ *= perron_frobenius
end
auto_correlation_snapshots .= auto_correlation_snapshots .- sum(val .* p)^2;
auto_correlation_snapshots .*= 1.0 / auto_correlation_snapshots[1];

##
auto_fig = Figure()
ax1 = Axis(auto_fig[1, 1]; title="Ensemble Statistics")
l1 = lines!(ax1, auto_correlation_snapshots[:], color=:red)
l2 = lines!(ax1, auto_correlation_timeseries[:], color=:blue)
Legend(auto_fig[1, 2], [l1, l2], ["Markov", "Timeseries"])
display(auto_fig)


##
auto_fig = Figure(resolution=(2700, 1800))
xfig = auto_fig[1, 1] = GridLayout()
yfig = auto_fig[2, 1] = GridLayout()
zfig = auto_fig[3, 1] = GridLayout()
koopman1_fig = auto_fig[1, 2] = GridLayout()
koopman2_fig = auto_fig[2, 2] = GridLayout()
z10_fig = auto_fig[3, 2] = GridLayout()
subfigs = [xfig, yfig, zfig, koopman1_fig, koopman2_fig, z10_fig]
colors = [:red, :blue, :orange]
labels = ["x", "y", "z", "koopman1", "koopman2", "1_{x <= 0}"]

reaction_coordinates = [u -> u[i] for i in 1:3] # define anonymous functions for reaction coordinates
reaction_coordinates = [reaction_coordinates..., [u -> real(iV[i, argmin([norm(u - s) for s in markov_states])]) for i in [2, 1]]..., u -> (u[1] <= 0)]
# reaction_coordinates = [u -> u[1], u -> u[2], u -> u[3], u -> real(iV[2, argmin([norm(u - s) for s in markov_states])]), u -> real(iV[1, argmin([norm(u - s) for s in markov_states])]), u -> (u[3] < 10)]

kwargs = (; ylabel="Autocorrelation", titlesize=30, ylabelsize=40, 
    xgridstyle=:dash, ygridstyle=:dash, ygridwidth=5, xgridwidth = 5,  xtickalign=1,
    xticksize=20, ytickalign=1, yticksize=20, xlabel = "Time", 
    xticklabelsize=40, yticklabelsize=40, xlabelsize = 40)


for i in ProgressBar(1:6)
    current_reaction_coordinate = reaction_coordinates[i]
    subfig = subfigs[i]

    markov = [current_reaction_coordinate(snapshot) for snapshot in snapshots]
    rtimeseries = [current_reaction_coordinate(state) for state in state_timeseries]

    total = 800
    auto_correlation_timeseries = zeros(total)
    for s in ProgressBar(0:total-1)
        auto_correlation_timeseries[s+1] = mean(rtimeseries[s+1:end] .* rtimeseries[1:end-s])
    end
    auto_correlation_timeseries .-= mean(rtimeseries)^2
    auto_correlation_timeseries .*= 1 / auto_correlation_timeseries[1]

    auto_correlation_snapshots = zeros(total)
    val = [current_reaction_coordinate(snapshot) for snapshot in snapshots]

    Pτ = perron_frobenius * 0 + I
    for i in 0:total-1
        auto_correlation_snapshots[i+1] = sum(val' * Pτ * (p .* val))
        Pτ *= perron_frobenius
    end
    auto_correlation_snapshots .= auto_correlation_snapshots .- sum(val .* p)^2
    auto_correlation_snapshots .*= 1.0 / auto_correlation_snapshots[1]


    ax1 = Axis(subfig[1, 1]; title= "Variable =  " * labels[i], kwargs...)
    l1 = lines!(ax1, dt .* collect(0:total-1), auto_correlation_snapshots[:], color=:purple, label="Markov", linewidth = 5)
    l2 = lines!(ax1, dt .* collect(0:total-1), auto_correlation_timeseries[:], color=:black, label="Timeseries", linewidth = 5)
    axislegend(ax1, position=:rt, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=100, labelsize=40)
    display(auto_fig)
end
display(auto_fig)
