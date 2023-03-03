fig = Figure(resolution=(2000, 1500))
ax = LScene(fig[1:2, 1:2]; show_axis=false) # title = "Phase Space and Partitions", titlesize = 40)

last_index = length(tuple_state) - 1# 3000
plot_state = tuple_state[2:last_index+1]

starting_colors = [:black for i in 1:last_index]
plot_colors = Observable(starting_colors)
scatter!(ax, Tuple.(markov_states), color=[:red, :blue, :orange], markersize=40.0)
lines!(ax, plot_state, color=plot_colors)
rotate_cam!(ax.scene, (0, pi / 2 + π / 12, 0))

time_indices = 1:60*15*2

function change_function(time_index)
    phase = 2π / (60 * 15)
    rotate_cam!(ax.scene, (0, phase, 0))
    if time_index == 900
        plot_colors[] = colors[1:last_index]
    end
end

# record(change_function, fig, "lorenz_animation_attractor.mp4", time_indices; framerate=framerate)

##

timeseries_fig = Figure(resolution=(1800, 1800))
xfig = timeseries_fig[1, 1] = GridLayout()
yfig = timeseries_fig[2, 1] = GridLayout()
zfig = timeseries_fig[3, 1] = GridLayout()

kwargs = (; titlesize=30, ylabelsize=80,
    xgridstyle=:dash, ygridstyle=:dash, ygridwidth=5, xgridwidth=5, xtickalign=1,
    xticksize=20, ytickalign=1, yticksize=20,
    xticklabelsize=40, yticklabelsize=40, xlabelsize=80)

reaction_coordinates = [u -> u[1], u -> u[2], u -> u[3]]
labels = ["x", "y", "z"]
subfigs = [xfig, yfig, zfig]
total = 1000
state_timeseries = copy(state)
for i in ProgressBar(1:3)
    current_reaction_coordinate = reaction_coordinates[i]
    subfig = subfigs[i]
    rtimeseries = [current_reaction_coordinate(state) for state in state_timeseries]
    if i == 3
        ax1 = Axis(subfig[1, 1]; xlabel="Time", ylabel=labels[i], kwargs...)
        ylims!(ax1, (-1, 50))
    else
        ax1 = Axis(subfig[1, 1]; ylabel=labels[i], kwargs...)
        ylims!(ax1, (-30, 30))
    end
    l1 = lines!(ax1, dt .* collect(0:total-1), rtimeseries[1:total], color=:black, linewidth=5)
    display(timeseries_fig)
end
display(timeseries_fig)

save("lorenz_dynamics.png", timeseries_fig)
##
using Random, Distributions
function next_state(current_state_index::Int, cT)
    vcT = view(cT, :, current_state_index)
    u = rand(Uniform(0, 1))
    # choose a random uniform variable and decide next state
    # depending on where one lies on the line with respect to the probability 
    # of being found between given probabilities
    for i in eachindex(vcT)
        if u < vcT[i]
            return i
        end
    end

    return i
end

function generate(matrix, n, dt, initial_condition)
    if all(sum(matrix, dims=1) .≈ 1)
        P = matrix
    else
        P = exp(matrix * dt)
    end
    cP = cumsum(P, dims=1)
    markov_chain = zeros(Int, n)
    markov_chain[1] = initial_condition
    for i = 2:n
        markov_chain[i] = next_state(markov_chain[i-1], cP)
    end
    return markov_chain
end

generate(Q, n; dt=1, initial_condition=rand(1:size(Q)[1])) = generate(Q, n, dt, initial_condition)
##
markov_fig = Figure(resolution=(1800, 1800))
xfig = markov_fig[1, 1] = GridLayout()
yfig = markov_fig[2, 1] = GridLayout()
zfig = markov_fig[3, 1] = GridLayout()

kwargs = (; titlesize=30, ylabelsize=80,
    xgridstyle=:dash, ygridstyle=:dash, ygridwidth=5, xgridwidth=5, xtickalign=1,
    xticksize=20, ytickalign=1, yticksize=20,
    xticklabelsize=40, yticklabelsize=40, xlabelsize=80)

reaction_coordinates = [u -> u[1], u -> u[2], u -> u[3]]
labels = ["x", "y", "z"]
subfigs = [xfig, yfig, zfig]
total = 1000
Qmat = Q # Q[]
Random.seed!(125)
state_timeseries = generate(Qmat, total, dt=dt, initial_condition=3)
new_colors = [color_choices[i] for i in state_timeseries]
for i in ProgressBar(1:3)
    current_reaction_coordinate = reaction_coordinates[i]
    subfig = subfigs[i]
    rtimeseries = [current_reaction_coordinate(markov_states[state]) for state in state_timeseries]
    if i == 3
        ax1 = Axis(subfig[1, 1]; xlabel="Time", ylabel=labels[i], kwargs...)
        ylims!(ax1, (-1, 50))
    else
        ax1 = Axis(subfig[1, 1]; ylabel=labels[i], kwargs...)
        ylims!(ax1, (-30, 30))
    end
    l1 = scatter!(ax1, dt .* collect(0:total-1), rtimeseries[1:total], color=new_colors, linewidth=5)
    display(markov_fig)
end
display(markov_fig)
save("markov_dynamics.png", markov_fig)


##
auto_fig = Figure(resolution=(900, 2 * 900))
xfig = auto_fig[1, 1] = GridLayout()
yfig = auto_fig[2, 1] = GridLayout()
zfig = auto_fig[3, 1] = GridLayout()
subfigs = [xfig, yfig, zfig]
colors = [:red, :blue, :orange]
labels = ["x", "y", "z"]
state_timeseries = copy(state)
reaction_coordinates = [u -> u[i] for i in 1:3] # define anonymous functions for reaction coordinates

kwargs = (; ylabel="Autocorrelation", titlesize=30, ylabelsize=40,
    xgridstyle=:dash, ygridstyle=:dash, ygridwidth=5, xgridwidth=5, xtickalign=1,
    xticksize=20, ytickalign=1, yticksize=20, xlabel="Time",
    xticklabelsize=40, yticklabelsize=40, xlabelsize=40)


for i in ProgressBar(1:3)
    current_reaction_coordinate = reaction_coordinates[i]
    subfig = subfigs[i]

    rtimeseries = [current_reaction_coordinate(state) for state in state_timeseries]

    total = 800
    auto_correlation_timeseries = zeros(total)
    for s in ProgressBar(0:total-1)
        auto_correlation_timeseries[s+1] = mean(rtimeseries[s+1:end] .* rtimeseries[1:end-s])
    end
    auto_correlation_timeseries .-= mean(rtimeseries)^2
    auto_correlation_timeseries .*= 1 / auto_correlation_timeseries[1]


    ax1 = Axis(subfig[1, 1]; title="Variable = " * labels[i], kwargs...)
    l2 = lines!(ax1, dt .* collect(0:total-1), auto_correlation_timeseries[:], color=:black, label="Timeseries", linewidth=5)
    if i == 3
        ylims!(ax1, (-1.1, 1.1))
    else
        ylims!(ax1, (-0.1, 1.1))
    end
    # axislegend(ax1, position=:rt, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=100, labelsize=40)
    display(auto_fig)
end
display(auto_fig)

##

auto_fig = Figure(resolution=(3000, 2000))
xfig = auto_fig[1, 1] = GridLayout()
yfig = auto_fig[2, 1] = GridLayout()
zfig = auto_fig[3, 1] = GridLayout()

xfig2 = auto_fig[1, 2] = GridLayout()
yfig2 = auto_fig[2, 2] = GridLayout()
zfig2 = auto_fig[3, 2] = GridLayout()

subfigs = [xfig, yfig, zfig, xfig2, yfig2, zfig2]
colors = [:red, :blue, :orange]

labels = ["x", "y", "z"]
reaction_coordinates = [u -> u[i] for i in 1:3] # define anonymous functions for reaction coordinates

labels = [labels..., "x > 0", "y > 0", "z > 5"]
reaction_coordinates = [reaction_coordinates..., u -> u[1] > 0, u -> u[2] > 0, u -> u[3] > 5]

kwargs = (; ylabel="Autocorrelation", titlesize=30, ylabelsize=40,
    xgridstyle=:dash, ygridstyle=:dash, ygridwidth=5, xgridwidth=5, xtickalign=1,
    xticksize=20, ytickalign=1, yticksize=20, xlabel="Time",
    xticklabelsize=40, yticklabelsize=40, xlabelsize=40)


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


    ax1 = Axis(subfig[1, 1]; title="Variable =  " * labels[i], kwargs...)
    l1 = lines!(ax1, dt .* collect(0:total-1), auto_correlation_snapshots[:], color=:purple, label="Markov", linewidth=5)
    l2 = lines!(ax1, dt .* collect(0:total-1), auto_correlation_timeseries[:], color=:black, label="Timeseries", linewidth=5)
    axislegend(ax1, position=:rt, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=100, labelsize=40)
    display(auto_fig)
end
display(auto_fig)
save("autocorrelation.png", auto_fig)
##
hfig = Figure(resolution=(1000, 1500))
xfig = hfig[1, 1] = GridLayout()
yfig = hfig[2, 1] = GridLayout()
zfig = hfig[3, 1] = GridLayout()
subfigs = [xfig, yfig, zfig]
colors = [:red, :blue, :orange]
labels = ["x", "y", "z"]

# reaction_coordinate(u) = real(iV[1, argmin([norm(u - s) for s in markov_states])]) # u[3] # 
reaction_coordinates = [u -> u[i] for i in 1:3] # define anonymous functions for reaction coordinates
kwargs = (; ylabel="probability", titlesize=30, ylabelsize=40, xgridstyle=:dash, ygridstyle=:dash, xtickalign=1,
    xticksize=20, ytickalign=1, yticksize=20,
    xticklabelsize=40, yticklabelsize=40)
bins1 = 30
bins2 = 20
snapshots = copy(markov_states)
state_timeseries = copy(state)
for i in 1:3
    current_reaction_coordinate = reaction_coordinates[i]
    subfig = subfigs[i]

    markov = [current_reaction_coordinate(snapshot) for snapshot in snapshots]
    rtimeseries = [current_reaction_coordinate(state) for state in state_timeseries]
    xs_m, ys_m = histogram(markov, normalization=p, bins=bins1, custom_range=extrema(rtimeseries))
    xs_t, ys_t = histogram(rtimeseries, bins=bins2, custom_range=extrema(rtimeseries))


    # ax1 = Axis(subfig[1, 1]; title="Markov Chain Histogram, " * labels[i], kwargs...)
    ax2 = Axis(subfig[1, 1]; title="Timeseries Histogram, " * labels[i], kwargs...)
    #=
    for ax in [ax1, ax2]
        x_min = minimum([minimum(xs_m), minimum(xs_t)])
        x_max = maximum([maximum(xs_m), maximum(xs_t)])
        y_min = minimum([minimum(ys_m), minimum(ys_t)])
        y_max = maximum([maximum(ys_m), maximum(ys_t)])
        xlims!(ax, (x_min, x_max))
        ylims!(ax, (y_min, y_max))
    end
    =#
    # barplot!(ax1, xs_m, ys_m, color=:purple)
    barplot!(ax2, xs_t, ys_t, color=:black)
    # hideydecorations!(ax2, grid=false)


end
display(hfig)


##

function quick_test(μ; N=1000, p=1)
    x = collect(0:N) ./ N
    Δx = x[2:end] .- x[1:end-1]
    ρ = 2 .* x
    val = 0.0
    for i in eachindex(Δx)
        val += (Δx[i] * abs(x[i] - μ)^p * ρ[i]) / 2
        val += (Δx[i] * abs(x[i+1] - μ)^p * ρ[i+1]) / 2
    end
    return val
end
