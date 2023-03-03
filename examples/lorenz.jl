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
markov_states = [markov_states..., [-sqrt(72), -sqrt(72), 27], [0.0, 0.0, 0.0], [sqrt(72), sqrt(72), 27]]
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
snapshots = markov_states
state_timeseries = state
reaction_coordinate(u) = u[3] # norm(u) # - mean(u[4][1+shiftx:15+shiftx, 1+shifty:15+shifty, end-2], dims =(1,2))[1] .* mean(u[2][1+shiftx:15+shiftx, 1+shifty:15+shifty, end-2], dims=(1, 2))[1]  # u[2][120, 30, end-0] * u[4][120, 30, end-0] # mean(u[1][:, 30, end], dims=1)[1] # * u[4][140, 60, end-2] # * u[1][130, 60, end-1] # * u[4][140+0, 60+0, end]  # distance(u, snapshots[4])# real(iV[1, argmin([distance(u, s) for s in snapshots])])
markov = [reaction_coordinate(snapshot) for snapshot in snapshots]
rtimeseries = [reaction_coordinate(state) for state in state_timeseries]
xs_m, ys_m = histogram(markov, normalization=p, bins=20, custom_range=extrema(rtimeseries))
xs_t, ys_t = histogram(rtimeseries, bins=20, custom_range=extrema(rtimeseries))
fig = Figure()
kwargs = (; ylabel="probability", titlesize=30, ylabelsize=30)
ax1 = Axis(fig[1, 1]; title="Ensemble Statistics", kwargs...)
ax2 = Axis(fig[1, 2]; title="Temporal Statistics", kwargs...)
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
ensemble_mean = sum(p .* markov)
temporal_mean = mean(rtimeseries)
ensemble_variance = sum(p .* (markov .^ 2)) - sum(p .* markov)^2
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
total = 800*3*3
auto_correlation_timeseries = zeros(total)
for s in ProgressBar(0:total-1)
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
l1 = lines!(ax1, auto_correlation_snapshots[:], color=:red)
l2 = lines!(ax1, auto_correlation_timeseries[:], color=:blue)
Legend(auto_fig[1, 2], [l1, l2], ["Markov", "Timeseries"])
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