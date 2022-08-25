d1 = 10
d = d1^2
s = Vector{Float64}[]
push!(s, zeros(d))
γ = ones(d)
Δt = 0.01
for i in ProgressBar(1:100000)
    sⁿ = s[i]
    # sⁿ⁺¹ = sⁿ - sⁿ * Δt + √(2Δt) * randn(d)
    sⁿ⁺¹ = (sⁿ + √(2Δt) * randn(d)) ./ (1 .+ γ .* Δt)
    push!(s, sⁿ⁺¹)
end

s₁ = [s1[1] for s1 in s]
s₂ = [s2[2] for s2 in s]
r = norm.(s) ./ sqrt(d)
r∞ = norm.(s, Inf)
r1 = norm.(s, 1) ./ d
mean(s₁)
var(s₁)
mean(r)
log10(var(r) / mean(r))
-log10(d) - 0.3
mean(r) / var(r)
d * 2
var(r) / var(r1)
var(r∞) / var(r)
##
heatmap(reshape(s[end], (d1, d1)), colormap = :balance, colorrange = (-3,3))
##
function histogram(
    array;
    bins=minimum([100, length(array)]),
    normalization=:uniform,
    custom_range=false
)
    tmp = zeros(bins)
    if custom_range isa Tuple
        down, up = custom_range
    else
        down, up = extrema(array)
    end
    down, up = down == up ? (down - 1, up + 1) : (down, up) # edge case
    bucket = collect(range(down, up, length=bins + 1))
    if normalization == :uniform
        normalization = ones(length(array)) ./ length(array)
    end
    for i in eachindex(array)
        # normalize then multiply by bins
        val = (array[i] - down) / (up - down) * bins
        ind = ceil(Int, val)
        # handle edge cases
        ind = maximum([ind, 1])
        ind = minimum([ind, bins])
        tmp[ind] += normalization[i]
    end
    return (bucket[2:end] + bucket[1:(end-1)]) .* 0.5, tmp
end
##
xs₁, ys₁ = histogram(s₁, bins=20, custom_range=extrema(s₁))
xr, yr = histogram(r, bins=maximum([20, floor(Int, sqrt(d))]), custom_range=extrema(r[end-10000:end]))
##
fig = Figure()
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])
ax3 = Axis(fig[2, 1])
ax4 = Axis(fig[2, 2])
lines!(ax1, s₁[1:10000])
lines!(ax2, r[1:10000])
barplot!(ax3, xs₁, ys₁, color=:red)
barplot!(ax4, xr, yr, color=:blue)
xlims!(ax4, (0, maximum(r)))
display(fig)

##
# snapshots = [state for state in s[1:10000:end-1000]]
snapshots = [state for state in s[1000:10000:end-1000]]
# snapshots = [randn(d) for i in 1:100]
# snapshots ./= norm.(snapshots) # good partition but then need to select state

function distance(x, y)
    return norm(x - y)
end
# or the distances function
current_state = Int64[]
for i in ProgressBar(eachindex(s))
    state = s[i]
    partition = argmin([distance(state, snapshot) for snapshot in snapshots])
    push!(current_state, partition)
end

length(union(current_state)) == length(snapshots)

count_matrix = zeros(length(snapshots), length(snapshots));
for i in 1:length(current_state)-1
    count_matrix[current_state[i+1], current_state[i]] += 1
end

perron_frobenius = count_matrix ./ sum(count_matrix, dims=1)
backwards_frobenius = count_matrix' ./ sum(count_matrix', dims=1)
Q = transition_rate_matrix(current_state, length(snapshots); γ=Δt);
estimated_error = norm(exp(Q * Δt) - perron_frobenius) / norm(perron_frobenius)
Λ, V = eigen(Q)
iV = inv(V)

p = real.(V[:, end] / sum(V[:, end]))
entropy = sum(-p .* log.(p) / log(length(snapshots)))
println("The entropy is ", entropy) # uniform distribution for a given N is always assigned to be one
println("The temporal scales are ", abs(real(1 / Λ[1])), " to ", abs(real(1 / Λ[end-1])), " units")
##
function histogram(
    array;
    bins=minimum([100, length(array)]),
    normalization=:uniform,
    custom_range=false
)
    tmp = zeros(bins)
    if custom_range isa Tuple
        down, up = custom_range
    else
        down, up = extrema(array)
    end
    down, up = down == up ? (down - 1, up + 1) : (down, up) # edge case
    bucket = collect(range(down, up, length=bins + 1))
    if normalization == :uniform
        normalization = ones(length(array)) ./ length(array)
    end
    for i in eachindex(array)
        # normalize then multiply by bins
        val = (array[i] - down) / (up - down) * bins
        ind = ceil(Int, val)
        # handle edge cases
        ind = maximum([ind, 1])
        ind = minimum([ind, bins])
        tmp[ind] += normalization[i]
    end
    return (bucket[2:end] + bucket[1:(end-1)]) .* 0.5, tmp
end

##
validation_set_size = 100000
validation_set = zeros(d, validation_set_size)
Δt = 0.01
validation_set[:, 1] .= s[end]
for i in ProgressBar(1:validation_set_size-1)
    sⁿ = validation_set[:, i]
    sⁿ⁺¹ = sⁿ - sⁿ * Δt + √(2Δt) * randn(d)
    validation_set[:, i+1] .= sⁿ⁺¹
end
##
reaction_coordinate(u) = u[1] # real(iV[end-2, argmin([distance(u, s) for s in snapshots])])
markov = [reaction_coordinate(snapshot) for snapshot in snapshots]
rtimeseries = [reaction_coordinate(validation_set[:, i]) for i in 1:size(validation_set, 2)]
xs_m, ys_m = histogram(markov, normalization=p, bins=20, custom_range=extrema(rtimeseries))
xs_t, ys_t = histogram(rtimeseries, bins=20, custom_range=extrema(rtimeseries))
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
temporal_mean = mean(rtimeseries)
ensemble_variance = sum(p .* markov .^ 2) - sum(p .* markov)^2
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
total = 200
auto_correlation_timeseries = zeros(total)
for s in 0:total-1
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
l1 = lines!(ax1, auto_correlation_snapshots[1:200], color=:red)
l2 = lines!(ax1, auto_correlation_timeseries[1:200], color=:blue)
Legend(auto_fig[1, 2], [l1, l2], ["Markov", "Timeseries"])
display(auto_fig)