using JLD2, LinearAlgebra
# jlfile = jldopen("/Users/andresouza/Desktop/atmospheric_timeseries.jld2")
# jlfile = jldopen("/Users/andresouza/Desktop/atmos_state.jld2")
jlfile = jldopen("/Users/andresouza/Desktop/larger_atmospheric_data.jld2")
current_state = copy(jlfile["current_state"])
close(jlfile)

new_current_state = current_state[1:floor(Int, length(current_state) / 2)]
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
dt = 24.40731687271974
X = 40
λₜ = real.(1 / (log(Λ[end-1]) / ( dt)) * X / 86400)
println("the slowest decaying statistical scale is ", λₜ, " days")



sp = reverse(sortperm(p))
sp2 = sortperm(sp)
p[sp[1]]
p[sp[2]]

new_state = similar(current_state)
for i in eachindex(current_state)
    new_state[i] = sp2[current_state[i]]
end

count_matrix = zeros(maximum(current_state), maximum(current_state));
for i in 1:length(current_state)-1
    count_matrix[new_state[i+1], new_state[i]] += 1
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

scatter(p)

scatter(sort(real.(iV[end-1, :]))[2:end])
