using StatisticalNonlocality, Distributions, Random, LinearAlgebra
import StatisticalNonlocality: ou_transition_matrix
import StatisticalNonlocality: uniform_phase

mat1 = ou_transition_matrix(4)
mat2 = uniform_phase(5)

mat(ω) = (1-ω) * mat1 + ω * mat2

eigvals(mat(1.0))