using HDF5
import StatisticalNonlocality: leicht_newman, discrete_laplacian

filename = "/Users/andresouza/Desktop/Repositories/StatisticalNonlocality/" * "ks_medium_res.h5"
fid = h5open(filename, "r")
u = read(fid["u"])[:, 30000:end]

norm(u[:, 1] - u[:, 2], Inf)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, u[:, 1], color=:blue)
lines!(ax, u[:, 962], color=:red)

##
snapshots = 1000
ũ = u[:, 1:snapshots]

Δ = discrete_laplacian(length(u[:, 1]))
Δ[1, 1] -= 1
Δ[end, end] -= 1
Δ[1, end] += 1
Δ[end, 1] += 1
D = [norm((ũ[:, i] - ũ[:, j]), Inf) for i in 1:snapshots, j in 1:snapshots]
D²² = [norm(Δ * (ũ[:, i] - ũ[:, j]), 2) for i in 1:snapshots, j in 1:snapshots]
Dᶜ = [abs(norm(Δ * (ũ[:, i])) - norm(Δ * (ũ[:, j]))) for i in 1:snapshots, j in 1:snapshots]

minimal_state_temporal_distance = maximum([D[i, i+1] for i in 1:snapshots-1]) # minimal connectivity in time

threshold = 0.05
distance_threshold = quantile(D[:], threshold)

distance_threshold = minimal_state_temporal_distance
C = (D .<= distance_threshold)

F = leicht_newman(C)

Q = 0 * similar(C)
for (i, j) in enumerate(vcat(F...))
    Q[i, j] = 1
end
C̃ = Q * C * Q'
##
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

##
state = ũ
states = []

current_state = Int[]
push!(states, state[:, 1])
push!(state_counts, 1)
push!(current_state, 1)
distance(x, y) = norm(Δ * (x - y), 2)
D = [distance(state[:, i], state[:, j]) for i in 1:snapshots, j in 1:snapshots]
minimal_state_temporal_distance = maximum([D[i, i+1] for i in 1:snapshots-1]) # minimal connectivity in time
distance_threshold = minimal_state_temporal_distance
tic = time()
for i in 2:snapshots
    candidate_state = state[:, i]
    distances = [distance(candidate_state, s) for s in states]
    if all(distances .> distance_threshold)
        push!(states, candidate_state)
        push!(current_state, length(states))
    else
        push!(current_state, argmin(distances))
    end
end
println("time of for the simulation is ", time() - tic, " seconds")

count_matrix = zeros(length(states), length(states))
for i in 1:snapshots-1
    count_matrix[current_state[i+1], current_state[i]] += 1
end
perron_frobenius = count_matrix ./ sum(count_matrix, dims=1)

F = []
for jj in 1:length(states)
    member_in_time = [i for (i, s) in enumerate(current_state) if s == jj]
    push!(F, member_in_time)
end
##
lin_to_c(i; m=8) = ((i - 1) % m + 1, (i - 1) ÷ m + 1)
reshape(collect(1:20), (4, 5))

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

##
newF = leicht_newman(perron_frobenius .> 0)
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


##
ind = 12
fig = Figure()
ax1 = Axis(fig[1, 1])
lines!(ax1, mean(ũ[:, F[ind]], dims=2)[:], color=:red)
for i in eachindex(F[ind])
    lines!(ax1, ũ[:, F[ind][i]], color=:blue)
end
display(fig)