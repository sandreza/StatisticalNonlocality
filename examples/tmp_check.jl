using StatisticalNonlocality: discrete_laplacian
using MarkovChainHammer.Trajectory: generate
using MarkovChainHammer.Utils: autocovariance
function advection_matrix_central(N; Δx = 1.0, u = 1.0)
    A = zeros(N,N)
    for i in 1:N 
        A[(i-1)%N+1, (i-1*0)%N+1] = 1
        A[(i+1)%N+1, (i-1*0)%N+1] = -1
    end
    return u * A / 2Δx
end

function advection_matrix_upwind(N; Δx=1.0, u=1.0)
    A = zeros(N, N)
    for i in 1:N
        A[(i-1)%N+1, (i-1*0)%N+1] = (u + abs(u))/2
        A[i, i] = -u
        A[(i+1)%N+1, (i-1*0)%N+1] = (u - abs(u))/2
    end
    return A / Δx
end

function discrete_laplacian_periodic(N; Δx = 1.0, κ = 1.0)
    Δ = zeros(N, N)
    for i in 1:N
        Δ[i, i] = -2
        Δ[(i-1)%N+1, (i-1*0)%N+1] = 1
        Δ[(i+1)%N+1, (i-1*0)%N+1] = 1
    end
    return κ * Δ / (Δx^2)
end
##
N = 4
Δx = 2π / N

A = advection_matrix_central(N; Δx)
#=
A_u = advection_matrix_upwind(N; Δx)
A_u_anti = (A_u - A_u')/2
A_u_sym = (A_u + A_u')/2
=#
Δ = discrete_laplacian_periodic(N; Δx)
sort(abs.(imag.(eigvals(advection_matrix(N; Δx)))))
eigvals(discrete_laplacian_periodic(N; Δx))
c = π/2
κ = π^2/8 # π^2 /8 
A * c 
Δ * κ
Q = A * c + Δ * κ
PF = exp(Q * 0.1)
tmp = generate(PF, 100000)
lines(tmp[1:400])
atmp = autocovariance(tmp; timesteps = 100)


N = 100
Δx = 2π / N
A = advection_matrix_central(N; Δx)
Δ = discrete_laplacian_periodic(N; Δx)
sort(abs.(imag.(eigvals(advection_matrix(N; Δx)))))
eigvals(discrete_laplacian_periodic(N; Δx))
c = 1.0 # π / 2
κ = 1.0 # π^2 / 8
A * c
Δ * κ
Q = A * c + Δ * κ
eigvals(Q)
PF = exp(Q * 0.1)
tmp = generate(PF, 100000)
lines(tmp[1:400])
ntmp = autocovariance(tmp; timesteps=100)

fig = Figure()
ax = Axis(fig[1,1])
scatter!(ax, atmp[1:100]/ atmp[1])
scatter!(ax, ntmp[1:100] / ntmp[1], color=(:red, 0.5), linewidth=10)
display(fig)

##

simulation_parameters = nstate_channel()
simulation_parameters = nstate_ou_1D()
simulation_parameters = nstate_ou_2D()

