using ProgressBars, GLMakie, FFTW, LinearAlgebra

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

periodic_state = Vector{Float64}[]
s = [−1.3763610682134e1, −1.9578751942452e1, 27.0]
subdiv = 2^11
push!(state, s)
dt = 1.5586522107162 / subdiv
for i in ProgressBar(1:2 * subdiv)
    s = rk4(lorenz!, s, dt)
    push!(periodic_state, s)
end
times = range(0, dt * 2 * subdiv, length = 2 * subdiv)

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
scatter!(ax1, log.(abs.(fft(xs[1:subdiv])))[1:64] ./ log(10))
scatter!(ax2, log.(abs.(fft(ys[1:subdiv])))[1:64] ./ log(10))
scatter!(ax3, log.(abs.(fft(zs[1:subdiv])))[1:64] ./ log(10))
display(fig)


##
state = Vector{Float64}[]
current_state = Int64[]
s = [14.0, 20.0, 27.0]
subdiv = 1000
push!(state, s)
dt = 1.5586522107162 / subdiv
markov_states = periodic_state[1:2^5:2^11]
for i in ProgressBar(1:1000000)
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
lines(Tuple.(periodic_state), color=:blue, linewidth=30)
lines!(Tuple.(state[1:end]))
