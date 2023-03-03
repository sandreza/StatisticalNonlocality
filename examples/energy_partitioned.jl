##
# Different levels of coarse graining for PF 

count_matrix = zeros(length(states), length(states));
reduced_current_state = copy(current_state)
for i in 1:length(reduced_current_state)-1
    count_matrix[reduced_current_state[i+1], reduced_current_state[i]] += 1
end
perron_frobenius1 = count_matrix ./ sum(count_matrix, dims=1);

count_matrix .= 0.0
reduced_current_state = copy(current_state[1:2:end])
for i in 1:length(reduced_current_state)-1
    count_matrix[reduced_current_state[i+1], reduced_current_state[i]] += 1
end
perron_frobenius2 = count_matrix ./ sum(count_matrix, dims=1);

count_matrix .= 0.0
reduced_current_state = copy(current_state[1:4:end])
for i in 1:length(reduced_current_state)-1
    count_matrix[reduced_current_state[i+1], reduced_current_state[i]] += 1
end
perron_frobenius4 = count_matrix ./ sum(count_matrix, dims=1);

count_matrix .= 0.0
reduced_current_state = copy(current_state[1:8:end])
for i in 1:length(reduced_current_state)-1
    count_matrix[current_state[i+1], reduced_current_state[i]] += 1
end
perron_frobenius8 = count_matrix ./ sum(count_matrix, dims=1);

count_matrix .= 0.0
reduced_current_state = copy(current_state[1:16:end])
for i in 1:length(reduced_current_state)-1
    count_matrix[reduced_current_state[i+1], reduced_current_state[i]] += 1
end
perron_frobenius16 = count_matrix ./ sum(count_matrix, dims=1);

count_matrix .= 0.0
reduced_current_state = copy(current_state[1:32:end])
for i in 1:length(reduced_current_state)-1
    count_matrix[reduced_current_state[i+1], reduced_current_state[i]] += 1
end
perron_frobenius32 = count_matrix ./ sum(count_matrix, dims=1);

count_matrix .= 0.0
reduced_current_state = copy(current_state[1:64:end])
for i in 1:length(reduced_current_state)-1
    count_matrix[reduced_current_state[i+1], reduced_current_state[i]] += 1
end
perron_frobenius64 = count_matrix ./ sum(count_matrix, dims=1);

count_matrix .= 0.0
reduced_current_state = copy(current_state[1:128:end])
for i in 1:length(reduced_current_state)-1
    count_matrix[reduced_current_state[i+1], reduced_current_state[i]] += 1
end
perron_frobenius128 = count_matrix ./ sum(count_matrix, dims=1);

##
norm(perron_frobenius1^2 - perron_frobenius2) / norm(perron_frobenius1)
norm(perron_frobenius2^2 - perron_frobenius4) / norm(perron_frobenius4)
norm(perron_frobenius4^2 - perron_frobenius8) / norm(perron_frobenius8)
norm(perron_frobenius8^2 - perron_frobenius16) / norm(perron_frobenius16)

##
function partition_coordinate(state)
    return mean(u .^2) > 8
end
ctimeseries = [partition_coordinate(state[:,i]) for i in 1:size(state, 2)]
plot(ctimeseries[1:1000])
partition_quantiles = range(0, 1, length = 2)
estates = quantile.(Ref(ctimeseries), partition_quantiles)

current_state = Int64[]
global newtic = time()
for i in 1:snapshots
    candidate_state = ctimeseries[i]
    distances = [distance(candidate_state, s) for s in estates]
    push!(current_state, argmin(distances))
    if time() - newtic > 1
        println("simulation currently at ", i, " out of ", snapshots)
        println("This is ", i / snapshots * 100, " percent of the simuation")
        println("There are currently ", length(estates), " states")
        global newtic = time()
    end
end

count_matrix = zeros(length(estates), length(estates));
reduced_current_state = copy(current_state)
for i in 1:length(reduced_current_state)-1
    count_matrix[reduced_current_state[i+1], reduced_current_state[i]] += 1
end
perron_frobeniuse = count_matrix ./ sum(count_matrix, dims=1);
Qe = transition_rate_matrix(current_state, length(estates); γ=1);
ΛQe = eigvals(Qe)
ΛPFe = eigvals(perron_frobeniuse)
println(log(ΛPFe[end-1]))
println(ΛQe[end-1])
checkfiz = sum(sum(Qe .> 0, dims=1)[:] .<= 2) == length(estates)

println("The check for physicality is ", checkfiz)
##
totes = 1000
u²_markov = zeros(totes)
dt = 1.0
val = collect(estates)
Pτ = perron_frobeniuse * 0 + I
Λ, V = eigen(perron_frobeniuse)
p = real(V[:, end] ./ sum(V[:, end]))
for i in 0:totes-1
    # τ = i * dt
    # Pτ = real.(V * Diagonal(exp.(Λ .* τ)) * V⁻¹)
    accumulate = 0.0
    accumulate += sum(val' * Pτ * (p .* val))
    u²_markov[i+1] = accumulate
    Pτ *= perron_frobeniuse
end
u²_markov .= u²_markov .- sum(val .* p)^2
##
u²_timeseries = zeros(totes)
for s in 0:totes-1
    u²_timeseries[s+1] = mean(ctimeseries[s+1:end] .* ctimeseries[1:end-s])
end
u²_timeseries .-= mean(ctimeseries)^2
##
fig = Figure()
ax1 = Axis(fig[1, 1]; title="Ensemble Statistics")
l1 = lines!(ax1, u²_markov, color=:red)
l2 = lines!(ax1, u²_timeseries, color=:blue)
Legend(fig[1, 2], [l1, l2], ["Markov", "Timeseries"])
display(fig)

##
energy_partitioned_states = []
energy_partitioned_indices = []
for i in 1:length(estates)
    micro_state = ũ[:, current_state.==i]
    if size(micro_state)[2] > 10000
        micro_state = micro_state[:, 1:10:end]
    end
    num_states = maximum([ceil(Int, length(micro_state[1, :]) / 1000), 4])
    println("The number of states is ", num_states)
    kmr = kmeans(micro_state, num_states)
    km_current_state = kmr.assignments
    states = [kmr.centers[:, i] for i in 1:num_states]
    push!(energy_partitioned_indices, length(energy_partitioned_states)+1:length(energy_partitioned_states)+num_states)
    push!(energy_partitioned_states, states...)
end

##
ecurrent_state = Int64[]
global newtic = time()
snapshot_estate = zeros(length(energy_partitioned_states[1]), length(energy_partitioned_states))
for i in 1:snapshots
    candidate_state = state[:, i]
    # macro state
    ecandidate_state = partition_coordinate(candidate_state) 
    eindex = energy_partitioned_indices[argmin([distance(ecandidate_state, s) for s in estates])]
    # micro state
    distances = [distance(candidate_state, s) for s in energy_partitioned_states[eindex]]
    push!(ecurrent_state, eindex[argmin(distances)])
    snapshot_estate[:, eindex[argmin(distances)]] .= state[:, i]
    if time() - newtic > 1
        println("simulation currently at ", i, " out of ", snapshots)
        println("This is ", i / snapshots * 100, " percent of the simuation")
        println("There are currently ", length(energy_partitioned_states), " states")
        global newtic = time()
    end
end
##
count_matrix = zeros(length(energy_partitioned_states), length(energy_partitioned_states))
reduced_current_state = copy(ecurrent_state[1:end])
for i in 1:length(reduced_current_state)-1
    count_matrix[reduced_current_state[i+1], reduced_current_state[i]] += 1
end
perron_frobenius_e = count_matrix ./ sum(count_matrix, dims=1);
Q_e = transition_rate_matrix(ecurrent_state, length(energy_partitioned_states); γ=1);
ΛQ_e = eigvals(Q_e)
ΛPF_e = eigvals(perron_frobenius_e)
log(ΛPF_e[end-1])
ΛQ_e[end-1]
##
ll, vv = eigen(Q_e)
p = real.(vv[:, end] ./ sum(vv[:, end]))
entropy = sum(-p .* log.(p) / log(length(energy_partitioned_states)))
println("The entropy is ", entropy) # uniform distribution for a given N is always assigned to be one
ll[end-1]
##
reaction_coordinate(u) = maximum(u) # argmin([distance(x, s) for s in energy_partitioned_states])
markov = [reaction_coordinate(snapshot_estate[:, i]) for i in 1:length(energy_partitioned_states)]
timeseries = [reaction_coordinate(u[:, i]) for i in snapshots:size(u)[2]]
xs_m, ys_m = histogram2(markov, normalization=p, bins=20, custom_range=extrema(timeseries))
xs_t, ys_t = histogram2(timeseries, bins=20, custom_range=extrema(timeseries))
fig = Figure()
ax1 = Axis(fig[1, 1]; title="Ensemble Statistics")
ax2 = Axis(fig[1, 2]; title="Temporal Statistics")
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

println("Checking the convergence of the statitics of the rms velocity")
ensemble_mean = sum(p .* markov)
temporal_mean = mean(timeseries)
ensemble_variance = sum(p .* markov .^ 2) - sum(p .* markov)^2
temporal_variance = mean(timeseries .^ 2) - mean(timeseries)^2
println("The ensemble mean is ", ensemble_mean)
println("The temporal mean is ", temporal_mean)
println("The null hypothesis is ", mean(markov))
println("The ensemble variance is ", ensemble_variance)
println("The temporal variance is ", temporal_variance)
println("The null hypothesis is ", var(markov))
println("The absolute error between the ensemble and temporal means is ", abs(ensemble_mean - temporal_mean))
println("The relative error between the ensemble and temporal variances are ", 100 * abs(ensemble_variance - temporal_variance) / temporal_variance, " percent")

##
Λ, V = eigen(Q_e)
V⁻¹ = inv(V)
p = real.(V[:, end] ./ sum(V[:, end], dims=1))
Λ[end-1]
##
totes = floor(Int64, 400 / skip)
u²_timeseries = zeros(totes)
for s in 0:totes-1
    u²_timeseries[s+1] = mean(timeseries[s+1:end] .* timeseries[1:end-s])
end
u²_timeseries .-= mean(timeseries)^2
u²_timeseries .*= 1 / u²_timeseries[1]
##
u²_markov = zeros(totes)
dt = 1.0
val = markov
# Pτ = perron_frobenius * 0 + I
for i in 0:totes-1
    τ = i * dt
    Pτ = real.(V * Diagonal(exp.(Λ .* τ)) * V⁻¹)
    accumulate = 0.0
    accumulate += sum(val' * Pτ * (p .* val))
    u²_markov[i+1] = accumulate
    # Pτ *= perron_frobenius
    if i % 10 == 0
        println("On iteration ", i)
    end
end
u²_markov .= u²_markov .- sum(val .* p)^2
u²_markov .*= 1 / u²_markov[1]
##
fig = Figure()
ax1 = Axis(fig[1, 1]; title="Ensemble Statistics")
l1 = lines!(ax1, u²_markov, color=:red)
l2 = lines!(ax1, u²_timeseries, color=:blue)
Legend(fig[1, 2], [l1, l2], ["Markov", "Timeseries"])
display(fig)

