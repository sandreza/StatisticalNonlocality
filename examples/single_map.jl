using Enzyme, Random, GLMakie, ProgressBars, Statistics, LinearAlgebra

function ulam_map(xⁿ)
    return 1 - 2 * xⁿ * xⁿ
end
function ulam_map⁻¹(xⁿ⁺¹)
    xⁿ = sqrt((1 - xⁿ⁺¹) / 2)
    return (-xⁿ, xⁿ)
end
ulam_prime(x) = autodiff(Forward, ulam_map, DuplicatedNoNeed, Duplicated(x, 1.0))[1]
ulam_prime(2.0)

xⁿ = 1 / 3
x = Float64[]
for i in ProgressBar(1:1000000)
    xⁿ⁺¹ = ulam_map(xⁿ)
    push!(x, xⁿ⁺¹)
    xⁿ = xⁿ⁺¹
end
hist(x, bins=100)

sL = 100
snapshots = range(-1, 1, length=sL)  #  reverse(cos.(range(0, π, length=sL))) #   
current_state = Int64[]
for snapshot in ProgressBar(x)
    push!(current_state, argmin([abs(s - snapshot) for s in snapshots]))
end

count_matrix = zeros(length(snapshots), length(snapshots));
for i in 1:length(current_state)-1
    count_matrix[current_state[i+1], current_state[i]] += 1
end
perron_frobenius = count_matrix ./ sum(count_matrix, dims=1)
Λ, V = eigen(perron_frobenius)
iV = inv(V)
p = real.(V[:, end] / sum(V[:, end]))
entropy = sum(-p .* log.(p) / log(length(snapshots)))
fig, _, _ = scatter(snapshots, p)

##
mean(x)
sum(p .* snapshots)
var(x)
sum(p .* snapshots .^ 2) - sum(p .* snapshots)^2

##
invmap⁻ = [ulam_map⁻¹(snapshot)[1] for snapshot in snapshots]
invmap⁺ = [ulam_map⁻¹(snapshot)[2] for snapshot in snapshots]
exact = 1.0 ./ abs.(ulam_prime.(invmap⁻)) + 1.0 ./ abs.(ulam_prime.(invmap⁺))
upper_bound = 1.0 ./ abs.(ulam_prime.(snapshots))
upper_bound[upper_bound .> 1.0] .= 1.0

rowsum = sum(perron_frobenius, dims=2)[:]
colprob = [maximum(perron_frobenius[:, i]) for i in 1:sL]
fig = Figure()
ax11 = Axis(fig[1, 1])
ax12 = Axis(fig[1, 2])
scatter!(ax11, snapshots, rowsum)
lines!(ax11, snapshots, exact, color = :red, linewidth = 3)
scatter!(ax12, snapshots, colprob)
lines!(ax12, snapshots, upper_bound, color = :red, linewidth = 3)
ylims!(ax12, (0, 1.2))
ylims!(ax11, (0, 10))
display(fig)
