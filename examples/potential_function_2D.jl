using Enzyme, Random, GLMakie, ProgressBars, Statistics, LinearAlgebra
import StatisticalNonlocality: transition_rate_matrix

function U(v⃗)
    x = v⃗[1]
    y = v⃗[2]
    return ((x - 1)^2 + 0.01) * ((x + 1)^2 + 0.0) * (x^2 + 0.02) + y^2
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

    k₁ = -∇U(xⁿ)
    x̃ = xⁿ + Δt * k₁ * 0.5
    k₂ = -∇U(x̃)
    x̃ = xⁿ + Δt * k₂ * 0.5
    k₃ = -∇U(x̃)
    x̃ = xⁿ + Δt * k₃
    k₄ = -∇U(x̃)

    𝒩 = randn(2)
    global xⁿ⁺¹ += Δt / 6 * (k₁ + 2 * k₂ + 2 * k₃ + k₄) + ϵ * sqrt(Δt) * 𝒩

    push!(x, xⁿ⁺¹)
    global xⁿ = xⁿ⁺¹

end

fig = Figure()
ax1 = Axis(fig[1,1])
ax2 = Axis(fig[1, 2])
hist!(ax1, [xc[1] for xc in x], bins=100)
hist!(ax2, [xc[2] for xc in x], bins=100)


#=
snapshots = [i for i in range(quantile.(Ref(x), [0.001, 0.999])..., length=200)]
# the dynamic allocation is faster than allocating memory for current_state 
# or the distances function
current_state = Int64[]
for snapshot in ProgressBar(x)
    push!(current_state, argmin([abs(s - snapshot) for s in snapshots]))
end

length(union(current_state)) == length(snapshots)
=#