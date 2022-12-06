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

fixed_points = [[-sqrt(72), -sqrt(72), 27], [0.0, 0.0, 0.0], [sqrt(72), sqrt(72), 27]]
markov_states = fixed_points

timeseries = Vector{Float64}[]
markov_chain = Int64[]
initial_condition = [14.0, 20.0, 27.0]
push!(timeseries, initial_condition)
dt = 0.01
iterations = 1000

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

# create colors for the plot
colors = Symbol[]
color_choices = [:red, :blue, :orange] # add more colors here for each partition
for i in eachindex(timeseries)
    push!(colors, color_choices[markov_chain[i]])
end
tuple_timeseries = Tuple.(timeseries)

# everything is done for plotting
fig = Figure(resolution = (1000, 700)) 
ax = LScene(fig[1:2,1:2]; show_axis = false)
lines!(ax, tuple_timeseries, color = colors)
rotate_cam!(ax.scene, (0, -π / 4, 0))
display(fig)