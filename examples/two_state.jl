using LinearAlgebra, StatisticalNonlocality
import StatisticalNonlocality: ou_transition_matrix

n = 1
M = ou_transition_matrix(n)

Λ, V = eigen(M)
Λ, W = eigen(M')
