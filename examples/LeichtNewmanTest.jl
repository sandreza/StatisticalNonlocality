using StatisticalNonlocality, LinearAlgebra, GLMakie, Random
import StatisticalNonlocality: leicht_newman
##
# Test the function
# number of clusters
Random.seed!(123)
nC = 5
# size of each cluster as a random number between 20 and 60
nS = rand(20:60, nC)
# total number of nodes
nT = sum(nS)
# adjacency matrix
A = zeros(Int, (nT, nT))
i = 0
for n in nS
    X = (rand(n, n) .< 2 / 3)
    A[i+1:i+n, i+1:i+n] = X
    global i += n
end
# permutation operator
p = shuffle(1:nT)
P = (I+zeros(nT, nT))[p, :]
# shuffled adjacency matrix
pA = P * A * P'

F = leicht_newman(pA)

Q = zeros(nT, nT)
for (i, j) in enumerate(vcat(F...))
    Q[i, j] = 1
end
C = Q * pA * Q'
println("Cluster labels change, but clusters don't")

fig = Figure()
ax1 = Axis(fig[1, 1]; title="Before")
ax2 = Axis(fig[1, 2]; title="Scrambled")
ax3 = Axis(fig[1, 3]; title="After")
heatmap!(ax1, A, colormap=Reverse(:grays))
heatmap!(ax2, pA, colormap=Reverse(:grays))
heatmap!(ax3, C, colormap=Reverse(:grays))
ax1.yreversed = true
ax2.yreversed = true
ax3.yreversed = true
display(fig)

##
# initialize
