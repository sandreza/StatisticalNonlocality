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

# AB periodic orbit
s = [−1.3763610682134e1, −1.9578751942452e1, 27.0]
T = 1.5586522107162

periodic_state = Vector{Float64}[]
subdiv = 2^11
push!(periodic_state, s)
dt = T / subdiv
for i in ProgressBar(1:subdiv)
    s = rk4(lorenz!, s, dt)
    push!(periodic_state, s)
end
ab_periodic_state = copy(periodic_state)

# AAB periodic orbit 
s = [−1.2595115397689e1, −1.6970525307084e1, 27.0]
T = 2.3059072639398
periodic_state = Vector{Float64}[]
subdiv = 2^11
push!(periodic_state, s)
dt = T / subdiv
for i in ProgressBar(1:subdiv)
    s = rk4(lorenz!, s, dt)
    push!(periodic_state, s)
end
aab_periodic_state = copy(periodic_state)

# ABB periodic orbit
s = [−1.4426408025035e1, −2.1111230056994e1, 27.0]
T = 2.3059072639398

periodic_state = Vector{Float64}[]
subdiv = 2^11
push!(periodic_state, s)
dt = T / subdiv
for i in ProgressBar(1:subdiv)
    s = rk4(lorenz!, s, dt)
    push!(periodic_state, s)
end
abb_periodic_state = copy(periodic_state)


fixed_points = [[-sqrt(72), -sqrt(72), 27], [0.0, 0.0, 0.0], [sqrt(72), sqrt(72), 27]]
m = 3
markov_states = [fixed_points..., ab_periodic_state[1:2^m:end-1]..., aab_periodic_state[1:2^m:end-1]..., abb_periodic_state[1:2^m:end-1]...]

timeseries = Vector{Float64}[]
markov_chain = Int64[]
initial_condition = [14.0, 20.0, 27.0]
push!(timeseries, initial_condition)
dt = 0.01
iterations = 1000000

# grab first markov state
# note: replace the below thing with the appropriate "assignment" function 
markov_index = argmin([norm(initial_condition - markov_state) for markov_state in markov_states])
push!(markov_chain, markov_index)

# grab the rest
for i in ProgressBar(2:iterations)
    local state = rk4(lorenz!, timeseries[i-1], dt)
    push!(timeseries, state)
    # partition state space according to most similar markov state
    markov_index = argmin([norm(state - markov_state) for markov_state in markov_states])
    push!(markov_chain, markov_index)
end
# markov_states = timeseries[randperm(length(timeseries))[1:100]]

##
# create colors for the plot
colors = []
# non-custom, see https://docs.juliaplots.org/latest/generated/colorschemes/
color_choices = cgrad(:rainbow, length(markov_states), categorical=true)[randperm(length(markov_states))]# 12 is the number of states
n = length(ab_periodic_state[1:2^m:end-1])
color_choices = [[:pink for i in 1:3]..., [:orange for i in 1:n]..., [:blue for i in 1:n]..., [:green for i in 1:n]...]
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
# scatter!(ax, Tuple.(markov_states), color=:black, markersize=5.0)
lines!(ax, tuple_timeseries, color=colors)

rotate_cam!(ax.scene, (0, -π / 4, 0))
display(fig)

last_time_index = minimum([60 * 15 * 2, length(timeseries)])
time_indices = 1:last_time_index

display(fig)

function change_function(time_index)
    phase = 2π / (60 * 15)
    rotate_cam!(ax.scene, (0, phase, 0))
end

# record(change_function, fig, "lorenz_animation_attractor_periodic_orbits.mp4", time_indices; framerate=2*60)