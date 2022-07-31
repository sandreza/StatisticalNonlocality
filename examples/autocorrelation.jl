##
Λ, V = eigen(Q)
V⁻¹ = inv(V)
p = real.(V[: , end] ./ sum(V[: , end], dims = 1))
Λ[end-1]
##
reaction_coordinate(x) = minimum(x)
##
totes = floor(Int64, 1000 / skip)
ctimeseries = [reaction_coordinate(ũ[:, i]) for i in 1:size(ũ)[2]]
u²_timeseries = zeros(totes)
for s in 0:totes-1
    u²_timeseries[s+1] = mean(ctimeseries[s+1:end] .* ctimeseries[1:end-s])
end
u²_timeseries .-= mean(ctimeseries) ^ 2
##
u²_markov = zeros(totes)
dt = 1.0
val = [reaction_coordinate(state) for state in states]
# val = [reaction_coordinate(newstates[:, i]) for i in 1:400]
Pτ = perron_frobenius * 0 + I
for i in 0:totes-1
    # τ = i * dt
    # Pτ = real.(V * Diagonal(exp.(Λ .* τ)) * V⁻¹)
    accumulate = 0.0
    accumulate += sum(val' * Pτ * (p .* val))
    u²_markov[i+1] = accumulate
    Pτ *= perron_frobenius
end
u²_markov .= u²_markov .- sum(val .* p)^2
##
fig = Figure()
ax1 = Axis(fig[1, 1]; title="Ensemble Statistics")
l1 = lines!(ax1, u²_markov, color=:red)
l2 = lines!(ax1, u²_timeseries, color=:blue)
Legend(fig[1, 2], [l1, l2], ["Markov", "Timeseries"])
display(fig)