using HDF5, GLMakie, Statistics, LinearAlgebra, Clustering, FFTW
import StatisticalNonlocality: leicht_newman, discrete_laplacian, transition_rate_matrix

## load and visualize data
filename = "/Users/andresouza/Desktop/Repositories/StatisticalNonlocality/" * "ks_high_res.h5"
fid = h5open(filename, "r")
u = read(fid["u"])[:, 30000:end]
# viz 
fig = Figure()
ax1 = Axis(fig[1, 1]; title="timeseries", xlabel="space", ylabel="time")
ax2 = Axis(fig[1, 2]; title="snapshot", xlabel="space", ylabel="amplitude")
heatmap!(ax1, u[:, 1:10000], colorrange=(-8.5, 8.5), colormap=:balance)
lines!(ax2, u[:, 1], colorrange=(-8.5, 8.5), color=:red)
##
# Partition Data Set
number_of_snapshots = 100000
training_set = u[:, 1:number_of_snapshots]
validation_set = u[:, number_of_snapshots+1:end]
##
# Determine Snapshots by Reducing data by a factor of 10000
snapshots = [training_set[:, i] for i in 1:10000:size(training_set, 2)]
# Distance Function 
distance(snapshot1, snapshot2) = norm(snapshot1 - snapshot2)
# Viz 
lines(snapshots[1])
lines!(snapshots[end])

##
# Train
current_state = Int64[]
tic = time()
for i in 1:size(training_set, 2)
    snapshot = training_set[:, i]
    distances = [distance(snapshot, s) for s in snapshots]
    push!(current_state, argmin(distances))
    toc = time()
    if toc - tic > 1
        println("currently at ", i, " out of ", size(training_set, 2))
        tic = time()
    end
end
scatter(current_state)
##
# Construct Transition Rates 
count_matrix = zeros(length(snapshots), length(snapshots));
for i in 1:length(current_state)-1
    count_matrix[current_state[i+1], current_state[i]] += 1
end
perron_frobenius = count_matrix ./ sum(count_matrix, dims=1)
Q = transition_rate_matrix(current_state, length(snapshots); γ=1);
estimated_error = norm(exp(Q) - perron_frobenius) / norm(perron_frobenius)
Λ, V = eigen(Q)
p = real.(V[:, end] / sum(V[:, end]))
entropy = sum(-p .* log.(p) / log(length(snapshots)))
println("The entropy is ", entropy) # uniform distribution for a given N is always assigned to be one
##
# Validate
reaction_coordinate(u) = maximum(u)# argmin([distance(u, s) for s in snapshots]) # mean(u .^2)
markov = [reaction_coordinate(state) for state in snapshots]
timeseries = [reaction_coordinate(validation_set[:, i]) for i in 1:size(validation_set, 2)]
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
# Validation 2 
total = 200
auto_correlation_timeseries = zeros(total)
for s in 0:total-1
    auto_correlation_timeseries[s+1] = mean(timeseries[s+1:end] .* timeseries[1:end-s])
end
auto_correlation_timeseries .-= mean(timeseries)^2
auto_correlation_timeseries .*= 1 / auto_correlation_timeseries[1]

auto_correlation_snapshots = zeros(total)
val = [reaction_coordinate(snapshot) for snapshot in snapshots]
Pτ = perron_frobenius * 0 + I
for i in 0:total-1
    auto_correlation_snapshots[i+1] = sum(val' * Pτ * (p .* val))
    Pτ *= perron_frobenius
    if i % 10 == 0
        println("On iteration ", i)
    end
end
auto_correlation_snapshots .= auto_correlation_snapshots .- sum(val .* p)^2
auto_correlation_snapshots .*= 1.0 / auto_correlation_snapshots[1]
##
fig = Figure()
ax1 = Axis(fig[1, 1]; title="Ensemble Statistics")
l1 = lines!(ax1, auto_correlation_snapshots[:], color=:red)
l2 = lines!(ax1, auto_correlation_timeseries[:], color=:blue)
Legend(fig[1, 2], [l1, l2], ["Markov", "Timeseries"])
display(fig)
