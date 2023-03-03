using Enzyme, Random, GLMakie, ProgressBars, Statistics, LinearAlgebra
import StatisticalNonlocality: transition_rate_matrix

function U(v⃗)
    x = v⃗[1] # - 0.05 * v⃗[2]
    y = v⃗[2] # + 0.05 * v⃗[1]
    return ((x - 1)^2 + 0.01) * ((x + 1)^2 + 0.0) * (x^2 + 0.02) + ((y - 1)^2 + 0.00 * abs.(x+1)) * (y + 1)^2
end


# better to use reverse mode than forward mode here (for some reason)
∇U(x) = gradient(Enzyme.Reverse, U, x)

Random.seed!(12345)
x = Vector{Float64}[]
xⁿ = [0.0, 0.0]
xⁿ⁺¹ = [0.0, 0.0]
Δt = 0.03
ϵ = 0.25
# need better timestepping
for i in ProgressBar(1:10000000)
    # forward euler 
    #=
    𝒩 = randn()
    xⁿ⁺¹ = xⁿ + force(xⁿ) * Δt + ϵ * 𝒩 * √Δt
    push!(x, xⁿ⁺¹)
    xⁿ = xⁿ⁺¹
    =#
    ##
    # for stability, RK4

    k₁ = -∇U(xⁿ)
    x̃ = xⁿ + Δt * k₁ * 0.5
    k₂ = -∇U(x̃)
    x̃ = xⁿ + Δt * k₂ * 0.5
    k₃ = -∇U(x̃)
    x̃ = xⁿ + Δt * k₃
    k₄ = -∇U(x̃)

    𝒩 = randn(2)
    𝒩[2] *= 3
    global xⁿ⁺¹ += Δt / 6 * (k₁ + 2 * k₂ + 2 * k₃ + k₄) + ϵ * sqrt(Δt) * 𝒩

    push!(x, xⁿ⁺¹)
    global xⁿ = xⁿ⁺¹

end

fig = Figure()
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])
xc1 = [xc[1] for xc in x]
xc2 = [xc[2] for xc in x]
hist!(ax1, xc1, bins=100)
hist!(ax2, xc2, bins=100)

xL = 50# 50
yL = 50# 50
xp = range(quantile.(Ref(xc1), [0.01, 0.99])..., length=xL)
yp = range(quantile.(Ref(xc2), [0.01, 0.99])..., length=yL)
snapshots = [[xcs, ycs] for xcs in xp, ycs in yp]
snapshots = snapshots[:]

# or the distances function
current_state = Int64[]
for snapshot in ProgressBar(x)
    i = argmin([norm(xcs - snapshot[1]) for xcs in xp])
    j = argmin([norm(ycs - snapshot[2]) for ycs in yp])
    push!(current_state, i + xL * (j - 1))
end
length(union(current_state)) == length(snapshots)

count_matrix = zeros(length(snapshots), length(snapshots));
for i in 1:length(current_state)-1
    count_matrix[current_state[i+1], current_state[i]] += 1
end
problandscape = reshape(sum(count_matrix, dims=1), (xL, yL))
perron_frobenius = count_matrix ./ sum(count_matrix, dims=1)
backwards_frobenius = count_matrix' ./ sum(count_matrix', dims = 1)
Q = transition_rate_matrix(current_state, length(snapshots); γ=Δt);
estimated_error = norm(exp(Q * Δt) - perron_frobenius) / norm(perron_frobenius)
Λ, V = eigen(Q)
iV = inv(V)
p = real.(V[:, end] / sum(V[:, end]))
entropy = sum(-p .* log.(p) / log(length(snapshots)))
p_exact = @. exp(-2U(snapshots) / (ϵ^2))
p_exact = p_exact / sum(p_exact)
fig = Figure()
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])
heatmap!(ax1, reshape(p_exact, (xL, yL)), interpolate=true, colormap=:afmhot)
heatmap!(ax2, reshape(p, (xL, yL)), interpolate=true, colormap=:afmhot)
##
# compare accuracy of numerical solution 
sum(snapshots .* p)
sum(snapshots .* p_exact)
mean(x)

##
# 6 wells so 6 partitions
fig = Figure()
cs = [real.(V[:, end-i]) for i in 0:5]
cps = [real.(iV[end-i, :]) for i in 0:5]
topax = [Axis(fig[1, i]) for i in 1:6]
bottomax = [Axis(fig[2, i]) for i in 1:6]
# choose a colormap so that white is zero
for i in 1:6
    mcs = maximum(abs.(cs[i]))
    heatmap!(topax[i], reshape(cs[i], (xL, yL)), interpolate=true, colormap=:balance, colorrange=[-mcs, mcs])
    mcps = maximum(abs.(cps[i]))
    heatmap!(bottomax[i], reshape(cps[i], (xL, yL)), interpolate=true, colormap=:balance, colorrange=(-mcps, mcps))
end
display(fig)

##
# check reaction coordinate 
rtimeseries1 = [real(iV[end-1, state]) for state in current_state]
rtimeseries2 = [real(iV[end-3, state]) for state in current_state]

fig = Figure()
ax11 = Axis(fig[1,1])
ax21 = Axis(fig[2,1])
ax12 = Axis(fig[1,2])
ax22 = Axis(fig[2,2])
scatter!(ax11, xc1[1:1:25000], markersize=5)
scatter!(ax21, xc2[1:1:25000], markersize=5)
lines!(ax12, -rtimeseries1[1:1:25000], color=:red, linewidth=3)
lines!(ax22, -rtimeseries2[1:1:25000], color=:red, linewidth=3)

display(fig)

##