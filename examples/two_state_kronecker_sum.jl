using LinearAlgebra, StatisticalNonlocality
import StatisticalNonlocality: ou_transition_matrix
import StatisticalNonlocality: fourier
using Kronecker

n = 1
Q̃ = ou_transition_matrix(n)
Q = Q̃ ⊕ (Q̃ ⊕ Q̃)
