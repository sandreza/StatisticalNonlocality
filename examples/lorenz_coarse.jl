using ProgressBars, GLMakie, FFTW, LinearAlgebra, StatisticalNonlocality, Statistics

function lorenz!(ṡ, s)
    ṡ[1] = 10.0 * (s[2] - s[1])
    ṡ[2] = s[1] * (28.0 - s[3]) - s[2]
    ṡ[3] = s[1] * s[2] - (8/3) * s[3]
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
for i in ProgressBar(1:2 * subdiv)
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
for i in ProgressBar(1:2 * subdiv)
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
for i in ProgressBar(1:2 * subdiv)
    s = rk4(lorenz!, s, dt)
    push!(periodic_state, s)
end
abb_periodic_state = copy(periodic_state)

periodic_state = ab_periodic_state

times = range(0, dt * 2 * subdiv, length = 2 * subdiv+1)

pxs = [s[1] for s in periodic_state]
pys = [s[2] for s in periodic_state]
pzs = [s[3] for s in periodic_state]
pxs[1] - pxs[subdiv+1]
pys[1] - pys[subdiv+1]
pzs[1] - pzs[subdiv+1]

fig = Figure()
ax1 = Axis(fig[1,1])
ax2 = Axis(fig[1,2])
ax3 = Axis(fig[2,1])
lines!(ax1, times, pxs)
scatter!(ax1,  range(0, dt * subdiv - dt*subdiv/64, length= 64), pxs[1:2^5:subdiv], color = :red)
lines!(ax2, times, pys)
scatter!(ax2, range(0, dt * subdiv - dt * subdiv / 64, length=64), pys[1:2^5:subdiv], color=:red)
lines!(ax3, times, pzs)
scatter!(ax3, range(0, dt * subdiv - dt * subdiv / 64, length=64), pzs[1:2^5:subdiv], color=:red)
display(fig)


fig = Figure()
ax1 = Axis(fig[1,1])
ax2 = Axis(fig[1,2])
ax3 = Axis(fig[2,1])
scatter!(ax1, log.(abs.(fft(pxs[1:subdiv])))[1:64] ./ log(10))
scatter!(ax2, log.(abs.(fft(pys[1:subdiv])))[1:64] ./ log(10))
scatter!(ax3, log.(abs.(fft(pzs[1:subdiv])))[1:64] ./ log(10))
display(fig)


##
state = Vector{Float64}[]
current_state = Int64[]
s = [14.0, 20.0, 27.0]
subdiv = 200
push!(state, s)
dt = 1.5586522107162 / subdiv
markov_states = periodic_state[1:2^4:2^11]
# markov_states = [ab_periodic_state[1:2^4:2^11]..., abb_periodic_state[1:2^4:2^11]..., aab_periodic_state[1:2^4:2^11]...]
markov_states = [[-sqrt(72), -sqrt(72), 27], [0.0, 0.0, 0.0], [sqrt(72), sqrt(72), 27]]
for i in ProgressBar(1:4*1000000)
    s = rk4(lorenz!, s, dt)
    push!(current_state, argmin([norm(s - ms) for ms in markov_states]))
    push!(state, s)
end

xs = [s[1] for s in state]
ys = [s[2] for s in state]
zs = [s[3] for s in state]

fig = Figure()
ax1 = Axis(fig[1,1])
ax2 = Axis(fig[1,2])
ax3 = Axis(fig[2,1])
lines!(ax1, xs[1:1000])
# scatter!(ax1,  range(0, dt * subdiv - dt*subdiv/64, length= 64), xs[1:2^5:subdiv], color = :red)
lines!(ax2, ys[1:1000])
# scatter!(ax2, range(0, dt * subdiv - dt * subdiv / 64, length=64), ys[1:2^5:subdiv], color=:red)
lines!(ax3, zs[1:1000])
# scatter!(ax3, range(0, dt * subdiv - dt * subdiv / 64, length=64), zs[1:2^5:subdiv], color=:red)
display(fig)
##
colors = Symbol[]
color_choices = [:red, :blue, :orange]
for i in eachindex(current_state)
    push!(colors, color_choices[current_state[i]])
end
tuple_state = Tuple.(state)
##

fig = Figure(resolution = (2000, 1500)) 
ax = LScene(fig[1:2,1:2]) # title = "Phase Space and Partitions", titlesize = 40)
ax2 = Axis(fig[1,3]; title = "Markov Embedding ", titlesize = 30)
ax3 = Axis(fig[2,3]; title = "Generator", titlesize = 30) # perhaps call generator? 
lorenz_slider = Slider(fig[3, 1:2], range=3:1:1000000, startvalue=1000)
observable_index = lorenz_slider.value
plot_state = @lift(tuple_state[2:$observable_index+1])
plot_colors = @lift(colors[1:$observable_index])
lines!(ax, plot_state, color = plot_colors)
scatter!(ax, @lift($plot_state[end]), color = :black, markersize = 40.0)
scatter!(ax, Tuple.(markov_states), color = [:red, :blue, :orange], markersize = 40.0)

xaxis_scatter = @lift(1:$observable_index)
yaxis_scatter = @lift(current_state[1:$observable_index])
sc = scatter!(ax2, yaxis_scatter, color = plot_colors)
xlims!(ax2, (1, 1000))
ylims!(ax2, (0.5, 3.5))

ax2.yticklabelsize = 40
index_names = ["Negative Lobe", "Origin", "Positive Lobe"]
ax2.yticks = ([1, 2, 3], index_names)

display(fig)

Q = @lift(transition_rate_matrix($yaxis_scatter, 3; γ=dt))
# rv = @lift(check_this($observable_index))
observable_index[] = 2
string_index_end1 = @lift( ($observable_index > 1500) ? 4 : 3)
string_index_end2 = @lift(($observable_index > 200) ? 4 : 3)
locations = Tuple[]
for i in 1:3, j in 1:3
    push!(locations, (i, j))
    if i == j
        text!(ax3, (4-i, j), color = color_choices[i], textsize = 40.0, text = @lift(string($Q[j,i])[1:$string_index_end2])) 
    else
        text!(ax3, (4-i, j), color = color_choices[i], textsize = 40.0, text = @lift(string($Q[j,i])[1:$string_index_end1])) 
    end
end
xlims!(ax3, (0,4))
ylims!(ax3, (0,4))
hidedecorations!(ax3)
text!(ax3, (0, 2), color=:black, textsize=40.0, text = "Q = ")

rotate_cam!(ax.scene, (0, -π/4, 0))
framerate = 2 * 30
# current_state up to 2000 seems nice 
time_upper_bound = 2000

tmp = ones(Int, 1000) * time_upper_bound  # repeat for a little while
tmp2 = ones(Int, 2) * (length(tuple_state)-1)
time_indices = [collect(2:1:time_upper_bound)..., tmp..., collect(time_upper_bound:100:100000)..., tmp2...] # collect(10000:100:100000)..., tmp2...]

function change_function(time_index)
    observable_index[] = time_index
    phase = 2π / 1000
    rotate_cam!(ax.scene, (0, phase,  0))
    if observable_index[] < Inf # time_upper_bound
        xlims!(ax2, (1, observable_index[]))
    else
        xlims!(ax2, (1, time_upper_bound))
    end
end

# record(change_function, fig, "lorenz_animation_3.mp4", time_indices; framerate = framerate)

##
scatter(current_state[end-10000:end])
##
lines(Tuple.(periodic_state), color=:blue, linewidth=30)
lines!(Tuple.(state[1:end]))
##
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
r_perron_frobenius = count_matrix' ./ sum(count_matrix', dims =1)
Λ, V =  eigen(perron_frobenius)

Q = transition_rate_matrix(current_state, length(markov_states); γ=dt);
rQ = transition_rate_matrix(reverse(current_state), length(markov_states); γ=dt)
Λ, V =  eigen(Q)
iV = inv(V)
p = real.(V[:,end] ./ sum(V[:,end]))
entropy = sum(-p .* log.(p) / log(length(markov_states)))
println("The entropy is ", entropy) # uniform distribution for a given N is always assigned to be one

fbmat = r_perron_frobenius * perron_frobenius
Λᵗ, Vᵗ = eigen(fbmat)

scatter(1 ./ abs.(Λ[1:end-1]))
scatter(abs.(Λ))

mean(xs)
mean(ys)
mean(zs)

mean(pxs)
mean(pys)
mean(pzs)

##
Q = transition_rate_matrix(current_state, length(markov_states); γ=dt);
Λ, V = eigen(Q)
iV = inv(V)
p = real.(V[:, end] ./ sum(V[:, end]))
perron_frobenius = exp(Q * dt)

hfig = Figure(resolution = (1800, 1500))
xfig = hfig[1, 1] = GridLayout()
yfig = hfig[2, 1] = GridLayout()
zfig = hfig[3, 1] = GridLayout()
subfigs = [xfig, yfig, zfig]
colors = [:red, :blue, :orange]
labels = ["x", "y", "z"]

# reaction_coordinate(u) = real(iV[1, argmin([norm(u - s) for s in markov_states])]) # u[3] # 
reaction_coordinates =  [u -> u[i] for i in 1:3] # define anonymous functions for reaction coordinates
kwargs = (; ylabel="probability", titlesize=30, ylabelsize=40, xgridstyle=:dash, ygridstyle=:dash, xtickalign=1,
    xticksize=20, ytickalign=1, yticksize=20,
    xticklabelsize=40, yticklabelsize=40)
bins1 = 30
bins2 = 20
snapshots = copy(markov_states)
state_timeseries = copy(state)
for i in 1:3
    current_reaction_coordinate = reaction_coordinates[i]
    subfig = subfigs[i]

    markov = [current_reaction_coordinate(snapshot) for snapshot in snapshots]
    rtimeseries = [current_reaction_coordinate(state) for state in state_timeseries]
    xs_m, ys_m = histogram(markov, normalization=p, bins=bins1, custom_range=extrema(rtimeseries))
    xs_t, ys_t = histogram(rtimeseries, bins=bins2, custom_range=extrema(rtimeseries))


    ax1 = Axis(subfig[1, 1]; title="Markov Chain Histogram, " * labels[i], kwargs...)
    ax2 = Axis(subfig[1, 2]; title="Timeseries Histogram, " * labels[i], kwargs...)

    for ax in [ax1, ax2]
        x_min = minimum([minimum(xs_m), minimum(xs_t)])
        x_max = maximum([maximum(xs_m), maximum(xs_t)])
        y_min = minimum([minimum(ys_m), minimum(ys_t)])
        y_max = maximum([maximum(ys_m), maximum(ys_t)])
        xlims!(ax, (x_min, x_max))
        ylims!(ax, (y_min, y_max))
    end

    barplot!(ax1, xs_m, ys_m, color= :purple)
    barplot!(ax2, xs_t, ys_t, color=:black)
    hideydecorations!(ax2, grid=false)


end
display(hfig)

##
tmp = []
tmp_labels = []
for i in 1:3
    push!(tmp, u -> u[i] )
    push!(tmp_labels, labels[i])
end
for i in 1:3 
    for j in i:3
        push!(tmp, u -> u[i]*u[j])
        push!(tmp_labels, labels[i] * labels[j])
    end
end

for i in eachindex(tmp_labels)
    tmpf = tmp[i]
    # println(" ensemble average of $(tmp_labels[i]) = $(sum(tmpf.(markov_states) .* p))")
    # println(" temporal average of $(tmp_labels[i]) = $(mean(tmpf.(state_timeseries)))")
    println(" ensemble: ⟨$(tmp_labels[i])⟩ = $(sum(tmpf.(markov_states) .* p))")
    println(" temporal: ⟨$(tmp_labels[i])⟩ = $(mean(tmpf.(state_timeseries)))")
    println("--------------------------------------------")
end

##
# check the holding times 
ht = construct_holding_times(current_state, 3; γ = dt)
bins = [5, 20, 100]
color_choices = [:red, :blue, :orange] # same convention as before
index_names = ["Negative Lobe", "Origin", "Positive Lobe"]
hi = 1 #holding index
bin_index = 1 # bin index
labelsize = 40
options = (; xlabel="Time", ylabel="Probability", titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize = labelsize, yticklabelsize = labelsize)
fig = Figure(resolution = (2800, 1800)) 
for hi in 1:3, bin_index in 1:3
    ax = Axis(fig[hi, bin_index]; title= index_names[hi] * " Holding Times "  * ", " * string(bins[bin_index]) * " Bins", options...)
    holding_time_index = hi

    holding_time_limits = (0,ceil(Int, maximum(ht[holding_time_index]) ))
    holding_time, holding_time_probability = histogram(ht[holding_time_index]; bins = bins[bin_index], custom_range = holding_time_limits)

    barplot!(ax, holding_time, holding_time_probability, color=color_choices[hi], label = "Data")
    λ = 1/mean(ht[holding_time_index])

    Δholding_time = holding_time[2] - holding_time[1]
    exponential_distribution = @. (exp( - λ  * (holding_time - 0.5 *  Δholding_time)) - exp( - λ  * (holding_time + 0.5 *  Δholding_time)) ) 
    lines!(ax, holding_time, exponential_distribution, color = :black, linewidth = 3)
    scatter!(ax, holding_time, exponential_distribution, color = :black, markersize = 10, label = "Exponential")
    axislegend(ax, position = :rt, framecolor = (:grey, 0.5), patchsize = (50, 50), markersize = 100, labelsize = 40)
end
display(fig)

##
auto_fig = Figure(resolution=(2700, 1800))
xfig = auto_fig[1, 1] = GridLayout()
yfig = auto_fig[2, 1] = GridLayout()
zfig = auto_fig[3, 1] = GridLayout()
koopman1_fig = auto_fig[1, 2] = GridLayout()
koopman2_fig = auto_fig[2, 2] = GridLayout()
z10_fig = auto_fig[3, 2] = GridLayout()
subfigs = [xfig, yfig, zfig, koopman1_fig, koopman2_fig, z10_fig]
colors = [:red, :blue, :orange]
labels = ["x", "y", "z", "koopman1", "koopman2", "1_{x <= 0}"]

reaction_coordinates = [u -> u[i] for i in 1:3] # define anonymous functions for reaction coordinates
reaction_coordinates = [reaction_coordinates..., [u -> real(iV[i, argmin([norm(u - s) for s in markov_states])]) for i in [2, 1]]..., u -> (u[1] <= 0)]
# reaction_coordinates = [u -> u[1], u -> u[2], u -> u[3], u -> real(iV[2, argmin([norm(u - s) for s in markov_states])]), u -> real(iV[1, argmin([norm(u - s) for s in markov_states])]), u -> (u[3] < 10)]

kwargs = (; ylabel="Autocorrelation", titlesize=30, ylabelsize=40, 
    xgridstyle=:dash, ygridstyle=:dash, ygridwidth=5, xgridwidth = 5,  xtickalign=1,
    xticksize=20, ytickalign=1, yticksize=20, xlabel = "Time", 
    xticklabelsize=40, yticklabelsize=40, xlabelsize = 40)


for i in ProgressBar(1:6)
    current_reaction_coordinate = reaction_coordinates[i]
    subfig = subfigs[i]

    markov = [current_reaction_coordinate(snapshot) for snapshot in snapshots]
    rtimeseries = [current_reaction_coordinate(state) for state in state_timeseries]

    total = 800
    auto_correlation_timeseries = zeros(total)
    for s in ProgressBar(0:total-1)
        auto_correlation_timeseries[s+1] = mean(rtimeseries[s+1:end] .* rtimeseries[1:end-s])
    end
    auto_correlation_timeseries .-= mean(rtimeseries)^2
    auto_correlation_timeseries .*= 1 / auto_correlation_timeseries[1]

    auto_correlation_snapshots = zeros(total)
    val = [current_reaction_coordinate(snapshot) for snapshot in snapshots]

    Pτ = perron_frobenius * 0 + I
    for i in 0:total-1
        auto_correlation_snapshots[i+1] = sum(val' * Pτ * (p .* val))
        Pτ *= perron_frobenius
    end
    auto_correlation_snapshots .= auto_correlation_snapshots .- sum(val .* p)^2
    auto_correlation_snapshots .*= 1.0 / auto_correlation_snapshots[1]


    ax1 = Axis(subfig[1, 1]; title= "Variable =  " * labels[i], kwargs...)
    l1 = lines!(ax1, dt .* collect(0:total-1), auto_correlation_snapshots[:], color=:purple, label="Markov", linewidth = 5)
    l2 = lines!(ax1, dt .* collect(0:total-1), auto_correlation_timeseries[:], color=:black, label="Timeseries", linewidth = 5)
    axislegend(ax1, position=:rt, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=100, labelsize=40)
    display(auto_fig)
end
display(auto_fig)


##
function coarse_grain_operator(identified_states)
    operator = zeros(length(identified_states), sum(length.(identified_states)))
    for (i, states) in enumerate(identified_states)
        for state in states
            operator[i, state] = 1
        end
    end
    return operator
end

numind = floor(Int, (length(markov_states) - 3) / 3)
ab_indices = collect(1:numind)
aab_indices = ab_indices[end] .+ collect(1:numind)
abb_indices = aab_indices[end] .+ collect(1:numind)
identified_states = [ab_indices, aab_indices, abb_indices, [abb_indices[end] + 1], [abb_indices[end] + 2], [abb_indices[end] + 3]]

P = coarse_grain_operator(identified_states)
P⁺ = (P ./ sum(P, dims=2))' # Moore-Penrose pseudoinverse, pinv(P) also works
Q̂ = P * Q * P⁺