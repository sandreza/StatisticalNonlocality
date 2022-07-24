##
Λ, V = eigen(Q)
V⁻¹ = inv(V)
##
totes = 1000
u²_timeseries = zeros(totes)
u²_markov = zeros(totes)
for s in 1:totes
    for j in 1:64
        @inbounds u²_timeseries[s] += mean(u[j, 1+3000:20000] .* u[j, s+3000:20000+s-1]) / 64
    end
end
##
for i in 1:1:1000
    τ = i * dt
    Pτ = real.(V * Diagonal(exp.(Λ .* τ)) * V⁻¹)
    accumulate = 0.0
    for j in 1:64
        val = [state[j] for state in states]
        accumulate += sum(val' * Pτ * (p .* val)) / 64
    end
    u²_markov[i] = accumulate
end
##
fig = Figure()
ax1 = Axis(fig[1, 1]; title="Ensemble Statistics")
l1 = lines!(ax1, u²_markov, color=:red)
l2 = lines!(ax1, u²_timeseries, color=:blue)
Legend(fig[1, 2],[l1, l2], ["Markov", "Timeseries"])
display(fig)