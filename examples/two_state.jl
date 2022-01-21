using LinearAlgebra, StatisticalNonlocality
import StatisticalNonlocality: ou_transition_matrix
import StatisticalNonlocality: fourier

filename = "two_state.jld2"
n = 1
M = ou_transition_matrix(n)

Λ, V = eigen(M)
Λ, W = eigen(M')

N = 128

D, x = fourier(N, a = 0, b = 2π)

γ = 1.0
U = 1.0
κ = 1.0

fluxkernel = -U * inv(κ .* D^2 - γ * I) * U
λ = eigvals(fluxkernel)
σ = svdvals(fluxkernel)

data_directory = "data"
mkpath(data_directory)
filepath = data_directory * "/" * filename
if isfile(filepath)
    rm(filepath)
end

file = jldopen(filepath, "a+")

file["kernel"] = fluxkernel
file["eigenvalues"] = λ
file["singularvalues"] = σ

close(file)



