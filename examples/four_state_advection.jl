using StatisticalNonlocality, LinearAlgebra, FFTW, SparseArrays, JLD2
import StatisticalNonlocality: chebyshev, fourier_nodes, fourier_wavenumbers
import StatisticalNonlocality: droprelativezeros!

# minimal configuration is M = 64, N = 8
N = 8 * 4
M = 8 * 4

# The first step is to build the block operators
a, b = 0, 4π
k = fourier_wavenumbers(N, L=b - a)
x = fourier_nodes(N, a=a, b=b)
Dz, z = chebyshev(M)

function build_operator_dirichlet(Dz, k)
    A = Dz * Dz .- k^2
    A[1, :] .= 0.0
    A[1, 1] = 1.0
    A[end, :] .= 0
    A[end, end] = 1.0
    return lu(A)
end
function build_operator_neuman(Dz, k)
    A = Dz * Dz .- k^2
    A[1, :] .= Dz[1,:]
    A[end, :] = Dz[end, :]
    return lu(A)
end


lu_fact = [build_operator_neumann(Dz, k[i]) for i in 1:round(Int,N/2)]

rhs = ones(33, 1000)
rhs[1, :] .= 0.0
rhs[end, :] .= 0.0
sol = lu_fact[2] \ rhs