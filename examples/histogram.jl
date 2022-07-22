
"""
histogram(array; bins = 100)
# Description
return arrays for plotting histogram
"""
function histogram(
    array;
    bins=minimum([100, length(array)]),
    normalize=true
)
    tmp = zeros(bins)
    down, up = extrema(array)
    down, up = down == up ? (down - 1, up + 1) : (down, up) # edge case
    bucket = collect(range(down, up, length=bins + 1))
    normalization = normalize ? length(array) : 1
    for i in eachindex(array)
        # normalize then multiply by bins
        val = (array[i] - down) / (up - down) * bins
        ind = ceil(Int, val)
        # handle edge cases
        ind = maximum([ind, 1])
        ind = minimum([ind, bins])
        tmp[ind] += 1 / normalization
    end
    return (bucket[2:end] + bucket[1:(end-1)]) .* 0.5, tmp
end

function histogram2(
    array;
    bins=minimum([100, length(array)]),
    normalization=:uniform
)
    tmp = zeros(bins)
    down, up = extrema(array)
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
reaction_coordinate(x) = sin(x[1]) / (1 + x[8]^2)

ll, vv = eigen(Q)
p = real.(vv[:, end] ./ sum(vv[:, end]))
markov = [reaction_coordinate(state) for state in states]
timeseries = [reaction_coordinate(u[:, i]) for i in 1:size(u)[2]]
xs_m, ys_m = histogram2(markov, normalization=p, bins=20)
xs_t, ys_t = histogram2(timeseries, bins=20)
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
println("The ensemble variance is ", ensemble_variance)
println("The temporal variance is ", temporal_variance)
println("The absolute error between the ensemble and temporal means is ", abs(ensemble_mean - temporal_mean))
println("keep in mind that this should be zero")
println("The relative error between the ensemble and temporal variances are ", 100 * abs(ensemble_variance - temporal_variance) / temporal_variance, " percent")