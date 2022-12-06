using ProgressBars, LinearAlgebra, Statistics
using GLMakie

# generate data
function lorenz!(ṡ, s)
    ṡ[1] = 10.0 * (s[2] - s[1])
    ṡ[2] = s[1] * (28.0 - s[3]) - s[2]
    ṡ[3] = s[1] * s[2] - (8 / 3) * s[3]
    return nothing
end

function rk4(f, s, dt)
    ls = length(s)
    k1 = zeros(ls)
    k2 = zeros(ls)
    k3 = zeros(ls)
    k4 = zeros(ls)
    f(k1, s)
    f(k2, s + k1 * dt / 2)
    f(k3, s + k2 * dt / 2)
    f(k4, s + k3 * dt)
    return s + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
end

# first define assignment function and some helper functions
x_partition(state) = state[1] < 0 ? 0 : 1
y_partition(state) = state[2] < 0 ? 0 : 1
function z_partition(state)
    if state[3] < 16
        return 0
    elseif state[3] < 30
        return 1
    else
        return 2
    end
end

function assignment(state)
    markov_index = 1
    markov_index += x_partition(state)
    markov_index += 2 * y_partition(state) # since 2 possible states in x
    markov_index += 2 * 2 * z_partition(state) # since 4 possible states in x and y
    return markov_index
end

timeseries = Vector{Float64}[]
markov_chain = Int64[]
initial_condition = [14.0, 20.0, 27.0]
push!(timeseries, initial_condition)
dt = 0.01
iterations = 100000

# grab first markov state
# note: replace the below thing with the appropriate "assignment" function 
markov_index = assignment(initial_condition)
push!(markov_chain, markov_index)

# grab the rest
for i in ProgressBar(2:iterations)
    local state = rk4(lorenz!, timeseries[i-1], dt)
    push!(timeseries, state)
    # partition state space according to most similar markov state
    markov_index = assignment(state)
    push!(markov_chain, markov_index)
end

# create colors for the plot
colors = []
# non-custom, see https://docs.juliaplots.org/latest/generated/colorschemes/
color_choices = cgrad(:rainbow, 12, categorical=true) # 12 is the number of states
# custom 
# color_choices = [RGBAf(1, 0, 0), RGBAf(0, 0, 1), RGBAf(1, 1, 0)]
# color_choices = [color_choices..., RGBAf(0, 1, 0), RGBAf(1, 0, 1), RGBAf(0, 1, 1)]
# color_choices = [color_choices..., RGBAf(0.5, 0.5, 0.5), RGBAf(0.5, 0, 0), RGBAf(0, 0.5, 0)]
# color_choices = [color_choices..., RGBAf(0, 0, 0.5), RGBAf(0.5, 0, 0.5), RGBAf(0, 0.5, 0.5)]
for i in eachindex(timeseries)
    push!(colors, color_choices[markov_chain[i]])
end
tuple_timeseries = Tuple.(timeseries)

# everything is done for plotting
fig = Figure(resolution=(1000, 700))
ax = LScene(fig[1:2, 1:2]; show_axis=false)
lines!(ax, tuple_timeseries, color=colors)
rotate_cam!(ax.scene, (0, -π / 4, 0))
display(fig)

last_time_index = minimum([60 * 15 * 2, length(timeseries)])
time_indices = 1:last_time_index

function change_function(time_index)
    phase = 2π / (60 * 15)
    rotate_cam!(ax.scene, (0, phase, 0))
end

record(change_function, fig, "lorenz_animation_attractor_2.mp4", time_indices; framerate=60)