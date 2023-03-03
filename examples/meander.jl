using JLD2, GLMakie, LinearAlgebra, ProgressBars, Statistics, Random
using StatisticalNonlocality

global_path = "/Users/andresouza/Desktop/"
filename = "meander_markov_states.jld2"
jlfile = jldopen(global_path * filename, "r")

current_state = copy(jlfile["current_state"])
dt = copy(jlfile["dt"]) # in days
us = []
vs = []
ws = []
bs = []
for i in ProgressBar(eachindex(keys(jlfile["u"])))
    push!(us, jlfile["u"][string(i)])
    push!(vs, jlfile["v"][string(i)])
    push!(ws, jlfile["w"][string(i)])
    push!(bs, jlfile["b"][string(i)])
end

close(jlfile)

snapshots = []
for j in eachindex(us)
    push!(snapshots, [us[j], vs[j], ws[j], bs[j]])
end
##
ht = construct_holding_times(current_state, 100, γ=dt)
mht = mean.(ht)
vht = sqrt.(var.(ht))
δht = abs.(mht - vht) .+ dt
aht = (mht + vht) / 2

##
heatmap(us[1][:, :, end], colormap=:balance, colorrange=(-1, 1))

##
mfig = Figure()
ax11 = Axis(mfig[1, 1])
ax12 = Axis(mfig[1, 2])
ax21 = Axis(mfig[2, 1])
ax22 = Axis(mfig[2, 2])

sl = Slider(mfig[3, 1:2], range=1:length(us), startvalue=3)
tindex = sl.value
colorrange = (-0.5, 0.5)
# field = @lift(bs[$tindex][:, :, end] .- (mean(bs[$tindex][:, :, end])-mean(mb)) - mb)
field11 = @lift(us[$tindex][:, 1:100, end-2])
field12 = @lift(vs[$tindex][:, 1:100, end-2])
field21 = @lift(ws[$tindex][:, 1:100, end-2])
field22 = @lift(bs[$tindex][:, 1:100, end-2])
heatmap!(ax11, field11, colormap=:balance, colorrange=colorrange)
heatmap!(ax12, field12, colormap=:balance, colorrange=colorrange)
heatmap!(ax21, field21, colormap=:balance, colorrange=(-1e-3, 1e-3))
heatmap!(ax22, field22, colormap=:thermometer, colorrange=(0, 0.015))
display(mfig)

##
mfig = Figure()
ax11 = Axis(mfig[1, 1])
ax12 = Axis(mfig[1, 2])
ax21 = Axis(mfig[2, 1])
ax22 = Axis(mfig[2, 2])

sl = Slider(mfig[3, 1:2], range=1:length(us), startvalue=3)
tindex = sl.value
colorrange = (-0.5, 0.5)
# field = @lift(bs[$tindex][:, :, end] .- (mean(bs[$tindex][:, :, end])-mean(mb)) - mb)
field11 = @lift(us[$tindex][:, 1:100, end-2])
field12 = @lift(vs[$tindex][:, 1:100, end-2])
field21 = @lift(ws[$tindex][:, 1:100, end-2])
field22 = @lift(bs[$tindex][:, 1:100, end-2])
heatmap!(ax11, field11, colormap=:balance, colorrange=colorrange)
heatmap!(ax12, field12, colormap=:balance, colorrange=colorrange)
heatmap!(ax21, field21, colormap=:balance, colorrange=(-1e-3, 1e-3))
heatmap!(ax22, field22, colormap=:thermometer, colorrange=(0, 0.015))
display(mfig)
##

# All together
afig = Figure()
all_ax = []
for i in 1:10, j in 1:10
    linind = (i-1) + 10 * (j-1) + 1
    cax = Axis(afig[i, j], title = string(linind))
    hidedecorations!(cax)
    hidespines!(cax)
    push!(all_ax, cax)
    field = bs[linind][:, 10:90, end-2]
    heatmap!(cax, field, colormap=:thermometer, colorrange=(0, 0.015))
end

display(afig)


##
# intimidation
ifig = Figure()
#=
for 
ax11 = Axis(ifig[1, 1])
ax12 = Axis(ifig[1, 2])
ax21 = Axis(ifig[2, 1])
ax22 = Axis(ifig[2, 2])

sl = Slider(mfig[3, 1:2], range=1:length(us), startvalue=3)
tindex = sl.value
colorrange = (-0.5, 0.5)
# field = @lift(bs[$tindex][:, :, end] .- (mean(bs[$tindex][:, :, end])-mean(mb)) - mb)
field11 = @lift(us[$tindex][:, 1:100, end-2])
field12 = @lift(vs[$tindex][:, 1:100, end-2])
field21 = @lift(ws[$tindex][:, 1:100, end-2])
field22 = @lift(bs[$tindex][:, 1:100, end-2])
heatmap!(ax11, field11, colormap=:balance, colorrange=colorrange)
heatmap!(ax12, field12, colormap=:balance, colorrange=colorrange)
heatmap!(ax21, field21, colormap=:balance, colorrange=(-1e-3, 1e-3))
heatmap!(ax22, field22, colormap=:thermometer, colorrange=(0, 0.015))
display(mfig)
=#

##
stragglers = setdiff(1:maximum(current_state), union(current_state))

current_state = [current_state..., stragglers..., current_state[1]]
count_matrix = zeros(maximum(current_state), maximum(current_state));
for i in 1:length(current_state)-1
    count_matrix[current_state[i+1], current_state[i]] += 1
end

column_sum = sum(count_matrix, dims=1) .> 0;
row_sum = sum(count_matrix, dims=2) .> 0;
if all(column_sum[:] .== row_sum[:])
    reduced_count_matrix = count_matrix[column_sum[:], column_sum[:]]
else
    println("think harder")
end

count_matrix = reduced_count_matrix

perron_frobenius = count_matrix ./ sum(count_matrix, dims=1)
backwards_frobenius = count_matrix' ./ sum(count_matrix', dims=1)
Λ, V = eigen(perron_frobenius)
iV = inv(V)
p = real.(V[:, end] / sum(V[:, end]))
entropy = sum(-p .* log.(p) / log(length(union(current_state))))
println("The timescales are ", -1 / log(abs(Λ[1])), " to ", -1 / log(abs(Λ[end-1])), " timesteps")
println(" which is ")
println("The timescales are ", -dt / log(abs(Λ[1])), " to ", -dt / log(abs(Λ[end-1])), " days")

##
lambda_fig = Figure()
ax1 = Axis(lambda_fig[1, 1]; xlabel="Index", ylabel=" Days", title="Eigenvalue Timescales", xlabelsize=30, ylabelsize=30, titlesize=30)
scatter!(ax1, -dt ./ log.(abs.(Λ[5:end-1])))
display(lambda_fig)


##
reaction_coordinate(u) = u[1][120, 60, end-1] * u[1][120+0, 60+0, end-4] # distance(u, snapshots[4])# real(iV[1, argmin([distance(u, s) for s in snapshots])])
markov = [reaction_coordinate(snapshot) for snapshot in snapshots]
# rtimeseries = [reaction_coordinate(validation_set[:, i]) for i in 1:size(validation_set, 2)]
xs_m, ys_m = histogram(markov, normalization=p, bins=20, custom_range=extrema(markov))
# xs_t, ys_t = histogram(rtimeseries, bins=20, custom_range=extrema(rtimeseries))
fig = Figure()
ax1 = Axis(fig[1, 1]; title="Ensemble Statistics")
# ax2 = Axis(fig[1, 2]; title="Temporal Statistics")
barplot!(ax1, xs_m, ys_m, color=:red)
# barplot!(ax2, xs_t, ys_t, color=:blue)
#=
for ax in [ax1, ax2]
    x_min = minimum([minimum(xs_m), minimum(xs_t)])
    x_max = maximum([maximum(xs_m), maximum(xs_t)])
    y_min = minimum([minimum(ys_m), minimum(ys_t)])
    y_max = maximum([maximum(ys_m), maximum(ys_t)])
    xlims!(ax, (x_min, x_max))
    ylims!(ax, (y_min, y_max))
end
=#
display(fig)
println("Checking the convergence of the statitics of the rms velocity")
ensemble_mean = sum(p .* markov)
# temporal_mean = mean(rtimeseries)
ensemble_variance = sum(p .* markov .^ 2) - sum(p .* markov)^2
# temporal_variance = mean(rtimeseries .^ 2) - mean(rtimeseries)^2
println("The ensemble mean is ", ensemble_mean)
# println("The temporal mean is ", temporal_mean)
println("The null hypothesis is ", mean(markov))
println("The ensemble variance is ", ensemble_variance)
# println("The temporal variance is ", temporal_variance)
println("The null hypothesis is ", var(markov))
# println("The absolute error between the ensemble and temporal means is ", abs(ensemble_mean - temporal_mean))
# println("The relative error between the ensemble and temporal variances are ", 100 * abs(ensemble_variance - temporal_variance) / temporal_variance, " percent")

##
spatial_correlation = Float64[]
for i in 1:200
    reaction_coordinate1(u) = u[1][140, 60, end-0]
    reaction_coordinate2(u) = u[1][i, 60, end-0]
    reaction_coordinate(u) = reaction_coordinate1(u) * reaction_coordinate2(u) # distance(u, snapshots[4])# real(iV[1, argmin([distance(u, s) for s in snapshots])])
    markov = [reaction_coordinate(snapshot) for snapshot in snapshots]
    markov1 = [reaction_coordinate1(snapshot) for snapshot in snapshots]
    markov2 = [reaction_coordinate2(snapshot) for snapshot in snapshots]
    ensemble_mean = sum(p .* markov) - sum(p .* markov1) * sum(p .* markov1)
    push!(spatial_correlation, ensemble_mean)
end

##
lines(spatial_correlation)