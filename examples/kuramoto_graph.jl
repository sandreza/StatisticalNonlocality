using HDF5, GLMakie, Statistics, LinearAlgebra, Clustering, FFTW
import StatisticalNonlocality: leicht_newman, discrete_laplacian, transition_rate_matrix

filename = "/Users/andresouza/Desktop/Repositories/StatisticalNonlocality/" * "ks_medium_res3.h5"
# filename = "/Users/andresouza/Desktop/Repositories/StatisticalNonlocality/" * "ks_high_res.h5"
fid = h5open(filename, "r")
skip = 1
u = read(fid["u"])[:, 30000:skip:end]

norm(u[:, 1] - u[:, 2], 2)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, u[:, 1], color=:blue)
lines!(ax, u[:, 2], color=:red)
display(fig)
##
snapshots = floor(Int64, 500000 / skip)
ũ = u[:, 1:snapshots]

##
state = ũ

maximum_states = Inf # Inf
states = []

current_state = Int[]
statesintime = Int[]
push!(statesintime, 1)
push!(states, state[:, 1])
push!(current_state, 1)
distance(x, y) = norm(x - y, 1)
# distance(x, y) = 0 * norm(x-y, 2) + 0.3 * norm( abs.(fft(x)) - abs.(fft(y)), 2)
# D = [distance(state[:, i], state[:, j]) for i in 1:snapshots, j in 1:snapshots]
function distance(x,y)
    return norm(x[1:16:end]-y[1:16:end], 2) # minimum([norm(circshift(x,i)- y) for i in 0:16:128])
end
temporal_distance = [distance(state[:, 1], state[:, i]) for i in 1:snapshots-1]
distance_threshold = quantile(temporal_distance, 0.1)
println("starting simulation")
tic = time()
newtic = time()
for i in 2:snapshots
    candidate_state = state[:, i]
    distances = [distance(candidate_state, s) for s in states]
    if length(states) < maximum_states
        if all(distances .>= distance_threshold)
            push!(states, candidate_state)
            push!(current_state, length(states))
        else
            push!(current_state, argmin(distances))
        end
    else
        push!(current_state, argmin(distances))
    end
    if time() - newtic > 1
        println("simulation currently at ", i, " out of ", snapshots)
        println("This is ", i / snapshots * 100, " percent of the simuation")
        println("There are currently ", length(states), " states")
        global newtic = time()
    end
    push!(statesintime, length(states))
end
println("There are currently ", length(states), " states")
println("time of for the simulation is ", time() - tic, " seconds")

lines(statesintime)
##
# for a set number of states
#=
current_state = Int64[]
global newtic = time()
for i in 1:snapshots
    candidate_state = state[:, i]
    distances = [distance(candidate_state, s) for s in states]
    push!(current_state, argmin(distances))
    if time() - newtic > 1
        println("simulation currently at ", i, " out of ", snapshots)
        println("This is ", i / snapshots * 100, " percent of the simuation")
        println("There are currently ", length(states), " states")
        global newtic = time()
    end
end
=#
##
num_states = 10
kmr = kmeans(ũ[:,1:100:end], num_states)
current_state = kmr.assignments
km_current_state = kmr.assignments
states = [kmr.centers[:, i] for i in 1:num_states]
##
current_state = Int64[]
global newtic = time()
for i in 1:snapshots
    candidate_state = state[:, i]
    distances = [distance(candidate_state, s) for s in states]
    push!(current_state, argmin(distances))
    if time() - newtic > 1
        println("simulation currently at ", i, " out of ", snapshots)
        println("This is ", i / snapshots * 100, " percent of the simuation")
        println("There are currently ", length(states), " states")
        global newtic = time()
    end
end


##
new_states = zeros(length(state[:,1]), length(states))
for i in 1:snapshots 
    new_states[:, current_state[i]] .= state[:, i]
end
sstates = []
for i in 1:length(states)
    push!(sstates, new_states[:, i])
end
states = sstates

##
state_norms = norm.(states)
state_norms_sorted = sortperm(state_norms)
energy_permutation = zeros(length(states), length(states))
for i in 1:length(states)
    energy_permutation[i, state_norms_sorted[i]] = 1.0
end
##

count_matrix = zeros(length(states), length(states));
for i in 1:snapshots-1
    count_matrix[current_state[i+1], current_state[i]] += 1
end
perron_frobenius = count_matrix ./ sum(count_matrix, dims=1);
Q = transition_rate_matrix(current_state, length(states); γ=1);
if length(states) < 2000
    percent_error = norm(exp(Q) - perron_frobenius) / (norm(perron_frobenius) + norm(exp(Q))) * 2 * 100
    fig, _, _ = scatter(reverse(abs.(eigvals(Q))))
    display(fig)
end

##
F = []
for jj in 1:length(states)
    member_in_time = [i for (i, s) in enumerate(current_state) if s == jj]
    push!(F, member_in_time)
end
println("The minimum number of states in a box is ", minimum(length.(F)))
##
reverse_sort = sortperm(state_norms_sorted)
Fsorted = F[state_norms_sorted]
perron_frobenius_sorted = energy_permutation * perron_frobenius * energy_permutation'
Qsorted = energy_permutation * Q * energy_permutation'
checkit = norm(Diagonal(Qsorted).diag[reverse_sort] - Diagonal(Q).diag)
# energy_sorted_states[state_norms_sorted[1]] ≂̸ states[1]
# but rather energy_sorted_states[reverse_sort[1]] = states[1] 
# energy_sorted_states[1] = states[state_norms_sorted[1]] 
# This says "state state_norms_sorted[1] gets mapped to energy sorted state 1"
# energy_permutation * norm.(states) = sorted energies, thing on left is out of order 
# 
# sort states by energy 
energy_sorted_states = states[state_norms_sorted]
# reverse sort here since we want to find indices to plug into the energy_sorted states
sorted_states = [reverse_sort[cs] for cs in current_state]
# which "energy sorted states" corresponds to which state
# that is to say states = energy_sorted_states[reverse_sort] 
energy_line = [norm(energy_sorted_states[ss]) for ss in sorted_states]
energy_line_check = [norm(states[s]) for s in current_state]
# cluster by energy quartiles?
# check on accuracy. Continuity of all soboleve norms demands that an ordered 
# thing should transfer to the thing in between. Gives an idea of the level of accuracy
energy_ordered = false
if energy_ordered == true
    states = copy(energy_sorted_states)
    F = copy(Fsorted)
    Q = copy(Qsorted)
    perron_frobenius = copy(perron_frobenius_sorted)
    current_state = copy(sorted_states)
end
energy_states = norm.(energy_sorted_states)
MM = 10
energy_partitions = [quantile(energy_states, i / MM) for i in 1:MM-1] # bad way to do it since not weigted by probability
starting_set = copy(energy_states)
starting_indices = 1:length(energy_states)
energy_partition_indices = []
for energy in energy_partitions
    set = starting_set .< energy
    push!(energy_partition_indices, starting_indices[set])
    starting_set = setdiff(starting_set, starting_set[set])
    starting_indices = setdiff(starting_indices, starting_indices[set])
end
push!(energy_partition_indices, starting_indices)
##

lin_to_c(i; m=8) = ((i - 1) % m + 1, (i - 1) ÷ m + 1)
reshape(collect(1:20), (4, 5))
if length(states) < 80
    fig = Figure()
    ax1 = Axis(fig[1, 1]; title="Total")
    heatmap!(ax1, ũ, colormap=:balance, colorrange=(-3, 3))
    axs = []
    for i in eachindex(F)
        jj, ii = lin_to_c(i)
        push!(axs, Axis(fig[ii, jj+1]; title="Group " * string(i)))
    end

    for (i, ax) in enumerate(axs)
        heatmap!(ax, ũ[:, F[i]], colormap=:balance, colorrange=(-3, 3))
    end
    display(fig)
end

##
start_value = 1
end_value = 10000

C = perron_frobenius .> 0.0
# newF = leicht_newman(C)
newF = energy_partition_indices
newclusters = [vcat(F[f]...) for f in newF]

lin_to_c(i; m=8) = ((i - 1) % m + 1, (i - 1) ÷ m + 1)

fig = Figure()
ax1 = Axis(fig[1, 1]; title="Total")
heatmap!(ax1, ũ[:, start_value:end_value], colormap=:balance, colorrange=(-3, 3))
axs = []
for i in eachindex(newclusters)
    jj, ii = lin_to_c(i)
    push!(axs, Axis(fig[ii, jj+1]; title="Group " * string(i)))
end

for (i, ax) in enumerate(axs)
    heatmap!(ax, ũ[:, newclusters[i]], colormap=:balance, colorrange=(-3, 3))
end
display(fig)

begin
    function cluster_function(i)
        argmax(in.(Ref(i), newF))
    end
end

#=
P = 0 * similar(C)
for (i, j) in enumerate(vcat(newF...))
    P[i, j] = 1
end
C̃ = P * C * P'
=#
##
start_value = 1 
end_value = 10000 
fig = Figure(resolution=(1832, 1448))
ax1 = Axis(fig[2, 1:3])
ax2 = Axis(fig[2, 4:6])
ax3 = Axis(fig[2, 7])

time_slider = Slider(fig[2, 8], range=start_value:end_value, startvalue=1, horizontal=false)
time_index = time_slider.value

makie_state = @lift(ũ[:, $time_index])
makie_state2 = @lift(states[current_state[$time_index]])
lines!(ax1, makie_state, color=:blue, linewidth=10)
lines!(ax1, makie_state2, color=:red)
lines!(ax2, makie_state2, color=:red)
uextrema = (-5,5)
ylims!(ax1, uextrema)
ylims!(ax2, uextrema)

heatmap!(ax3, ũ[:, start_value:end_value], colorrange=(-3, 3), colormap=:balance, interpolate=true)
hlines!(ax3, time_index, linewidth=10, color=:black)

observ_text1 = @lift("This is associated with markov state " * string(current_state[$time_index]))
Label(fig[1, 2], observ_text1, textsize=30)

# observ_text2 = @lift("This is markov state " * string(current_state[$time_index]) * " in cluster " * string(cluster_function(current_state[$time_index])) * " at time index " * string($time_index))
# Label(fig[1, 4], observ_text2, textsize=30)

display(fig)