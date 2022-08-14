using Enzyme, Random, GLMakie, ProgressBars, Statistics, LinearAlgebra
import StatisticalNonlocality: transition_rate_matrix

function potential(x)
    return ((x - 1)^2 + 0.01) * ((x + 1)^2 + 0.0) * (x^2 + 0.02)
end
force(x) = -autodiff(Forward, potential, DuplicatedNoNeed, Duplicated(x, 1.0))[1]

Random.seed!(12345)
x = Float64[]
xⁿ = 0.0
xⁿ⁺¹ = 0.0
Δt = 0.03 
ϵ = 0.25 
# need better timestepping
for i in ProgressBar(1:1000000)
    # forward euler 
    #=
    𝒩 = randn()
    xⁿ⁺¹ = xⁿ + force(xⁿ) * Δt + ϵ * 𝒩 * √Δt
    push!(x, xⁿ⁺¹)
    xⁿ = xⁿ⁺¹
    =#
    ##
    # for stability, RK4

    k₁ = force(xⁿ)
    x̃ = xⁿ + Δt * k₁ * 0.5
    k₂ = force(x̃)
    x̃ = xⁿ + Δt * k₂ * 0.5
    k₃ = force(x̃)
    x̃ = xⁿ + Δt * k₃
    k₄ = force(x̃)

    𝒩 = randn()
    global xⁿ⁺¹ += Δt / 6 * (k₁ + 2 * k₂ + 2 * k₃ + k₄) + ϵ * sqrt(Δt) * 𝒩

    push!(x, xⁿ⁺¹)
    global xⁿ = xⁿ⁺¹

end
hist(x, bins=100)

snapshots = [i for i in range(quantile.(Ref(x), [0.001, 0.999])..., length=200)]
# the dynamic allocation is faster than allocating memory for current_state 
# or the distances function
current_state = Int64[]
for snapshot in ProgressBar(x)
    push!(current_state, argmin([abs(s - snapshot) for s in snapshots]))
end

length(union(current_state)) == length(snapshots)

count_matrix = zeros(length(snapshots), length(snapshots));
for i in 1:length(current_state)-1
    count_matrix[current_state[i+1], current_state[i]] += 1
end
perron_frobenius = count_matrix ./ sum(count_matrix, dims=1)
Q = transition_rate_matrix(current_state, length(snapshots); γ=Δt);
estimated_error = norm(exp(Q * Δt) - perron_frobenius) / norm(perron_frobenius)
Λ, V = eigen(Q)
iV = inv(V)
p = real.(V[:, end] / sum(V[:, end]))
entropy = sum(-p .* log.(p) / log(length(snapshots)))
p_exact = @. exp(-2potential(snapshots) / (ϵ^2))
p_exact = p_exact / sum(p_exact)
fig, _, _ = scatter(p_exact)
scatter!(p)
display(fig)
##
# compare accuracy of numerical solution 
sum(snapshots .* p)
sum(snapshots .* p_exact)
mean(x)

sum((snapshots .^ 2) .* p) - sum(snapshots .* p)^2
sum((snapshots .^ 2) .* p_exact) - sum(snapshots .* p_exact)^2
mean(x .^ 2) - mean(x)^2
##
c1 = real.(V[:, end] * sign(V[1, end]))
cp1 = real.(iV[end, :]* sign(V[1, end]))
c2 = real.(V[:, end-1])
cp2 = real.(iV[end-1, :])
c3 = real.(V[:, end-2])
cp3 = real.(iV[end-2, :])
fig = Figure()
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])
ax3 = Axis(fig[1, 3])
scatter!(ax1, c1)
scatter!(ax1, cp1)
scatter!(ax2, c2)
scatter!(ax2, cp2)
scatter!(ax3, c3)
scatter!(ax3, cp3)
display(fig)
# lines(c1 - 1.2 * c3 + 0.12*c2)
##1# model reaction coordinate

g(a, b) = norm((real.(iV[end-1, :]) - (a * snapshots .+ b)))
as = range(-1, 1, length=101)
bs = range(-1, 1, length=101)
gs = [g(a, b) for a in as, b in bs]
ci = argmin(gs)
a = as[ci[1]]
b = bs[ci[2]]
rescale_x = a * x .+ b



rtimeseries = [real(iV[end-1, state]) for state in current_state]
# rte = maximum(abs.(rtimeseries))
# xe = maximum(abs.(x))
# rescale_x = rte[end] / xe[end] * x

fig, _, _ = lines(rtimeseries[1:10:100000], color=:red, linewidth=3)
scatter!(rescale_x[1:10:100000], markersize=5)
display(fig)

##
fig, _, _ = scatter(real.(iV[end-1, :]))
scatter!(real.(iV[end-2, :]))
