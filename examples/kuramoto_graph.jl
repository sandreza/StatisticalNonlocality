using HDF5
import StatisticalNonlocality: leicht_newman, discrete_laplacian

filename = "/Users/andresouza/Desktop/Repositories/StatisticalNonlocality/" * "ks_medium_res3.h5"
fid = h5open(filename, "r")
u = read(fid["u"])[:, 300000:4:end]

norm(u[:, 1] - u[:, 2], Inf)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, u[:, 1], color=:blue)
lines!(ax, u[:, 962], color=:red)

##
snapshots = 50000
ũ = u[:, 1:snapshots]

if snapshots < 1000
    Δ = discrete_laplacian(length(u[:, 1]))
    Δ[1, 1] -= 1
    Δ[end, end] -= 1
    Δ[1, end] += 1
    Δ[end, 1] += 1
    println("starting")
    D = [norm((ũ[:, i] - ũ[:, j]), Inf) for i in 1:snapshots, j in 1:snapshots]
    println("done with one")
    D²² = [norm(Δ * (ũ[:, i] - ũ[:, j]), 2) for i in 1:snapshots, j in 1:snapshots]
    println("done with two")
    Dᶜ = [abs(norm(Δ * (ũ[:, i])) - norm(Δ * (ũ[:, j]))) for i in 1:snapshots, j in 1:snapshots]
    println("done with three")

    minimal_state_temporal_distance = maximum([D[i, i+1] for i in 1:snapshots-1]) # minimal connectivity in time

    threshold = 0.05
    distance_threshold = quantile(D[:], threshold)

    distance_threshold = minimal_state_temporal_distance
    C = (D .<= distance_threshold)

    println("enacting algorithm")
    F = leicht_newman(C)

    Q = 0 * similar(C)
    for (i, j) in enumerate(vcat(F...))
        Q[i, j] = 1
    end
    C̃ = Q * C * Q'
end
##
if snapshots < 10000
    fig = Figure()
    ax1 = Axis(fig[1, 1]; title="Total")
    heatmap!(ax1, ũ, colormap=:balance, colorrange=(-3, 3))
    axs = []
    for i in eachindex(F)
        push!(axs, Axis(fig[1, i+1]; title="Group " * string(i)))
    end

    for (i, ax) in enumerate(axs)
        heatmap!(ax, ũ[:, F[i]], colormap=:balance, colorrange=(-3, 3))
    end
    display(fig)
end
##
state = ũ
states = []

current_state = Int[]
push!(states, state[:, 1])
push!(state_counts, 1)
push!(current_state, 1)
A = (I - 0.1 * Δ)
distance(x, y) = norm(x - y, 2)
# distance(x, y) = 0 * norm(x-y, 2) + 0.3 * norm( abs.(fft(x)) - abs.(fft(y)), 2)
# D = [distance(state[:, i], state[:, j]) for i in 1:snapshots, j in 1:snapshots]
temporal_distance = [distance(state[:, i], state[:, i+1]) for i in 1:snapshots-1]
minimal_state_temporal_distance = maximum(temporal_distance) # minimal connectivity in time
random_temporal_distance = mean([distance(state[:, i], state[:, rand(1:snapshots)]) for i in 1:snapshots-1]) # minimal connectivity in time
distance_threshold = 2.0 * minimal_state_temporal_distance + 0.0 * (random_temporal_distance - minimal_state_temporal_distance)
println("starting simulation")
tic = time()
newtic = time()
for i in 2:snapshots
    candidate_state = state[:, i]
    distances = [distance(candidate_state, s) for s in states]
    if all(distances .>= distance_threshold)
        push!(states, candidate_state)
        push!(current_state, length(states))
    else
        push!(current_state, argmin(distances))
    end
    if time() - newtic > 1
        println("simulation currently at ", i, " out of ", snapshots)
        println("This is ", i / snapshots * 100, " percent of the simuation")
        println("There are currently ", length(states), " states")
        global newtic = time()
    end
end
println("There are currently ", length(states), " states")
println("time of for the simulation is ", time() - tic, " seconds")


##
# for a set number of states
#=
current_state = []
states = [mean(ũ[:, f], dims=2)[:] for f in F]
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
num_states = 200
kmr = kmeans(ũ, num_states)
current_state = kmr.assignments
km_current_state = kmr.assignments
states = [kmr.centers[:, i] for i in 1:num_states]

##
state_norms = norm.(states)
state_norms_sorted = sortperm(state_norms)
energy_permutation = zeros(length(states), length(states))
for i in 1:length(states)
    energy_permutation[i, state_norms_sorted[i]] = 1.0
end
##

count_matrix = zeros(length(states), length(states))
for i in 1:snapshots-1
    count_matrix[current_state[i+1], current_state[i]] += 1
end
perron_frobenius = count_matrix ./ sum(count_matrix, dims=1)
Q = transition_rate_matrix(current_state, length(states); γ=1)
if length(states) < 2000
    percent_error = norm(exp(Q) - perron_frobenius) / (norm(perron_frobenius) + norm(exp(Q))) * 2 * 100
    fig, _, _ = scatter(reverse(abs.(eigvals(Q))))
    display(fig)
end


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
energy_ordered = true
if energy_ordered == true
    states = copy(energy_sorted_states)
    F = copy(Fsorted)
    Q = copy(Qsorted)
    perron_frobenius = copy(perron_frobenius_sorted)
    current_state = copy(sorted_states)
end


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
C = perron_frobenius .> 0.0
newF = leicht_newman(C)
newclusters = [vcat(F[f]...) for f in newF]

lin_to_c(i; m=8) = ((i - 1) % m + 1, (i - 1) ÷ m + 1)

fig = Figure()
ax1 = Axis(fig[1, 1]; title="Total")
heatmap!(ax1, ũ, colormap=:balance, colorrange=(-3, 3))
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

time_slider = Slider(fig[2, 8], range=start_value:end_value, startvalue=0, horizontal=false)
time_index = time_slider.value

makie_state = @lift(ũ[:, $time_index])
makie_state2 = @lift(states[current_state[$time_index]])
lines!(ax1, makie_state, color=:blue, linewidth=10)
lines!(ax1, makie_state2, color=:red)
lines!(ax2, makie_state2, color=:red)
ylims!(ax1, (-3, 3))
ylims!(ax2, (-3, 3))

heatmap!(ax3, ũ[:, start_value:end_value], colorrange=(-3, 3), colormap=:balance, interpolate=true)
hlines!(ax3, time_index, linewidth=10, color=:black)

observ_text1 = @lift("This is associated with markov state " * string(current_state[$time_index]))
Label(fig[1, 2], observ_text1, textsize=30)

observ_text2 = @lift("This is markov state " * string(current_state[$time_index]) * " in cluster " * string(cluster_function(current_state[$time_index])) * " at time index " * string($time_index))
Label(fig[1, 4], observ_text2, textsize=30)

display(fig)