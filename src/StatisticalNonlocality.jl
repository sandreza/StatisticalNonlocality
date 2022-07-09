module StatisticalNonlocality

# Transition Matrices for some Stochastic Processes
include("transition_matrices.jl")

# Chebyshev-Fourier Collocation method
include("spectral.jl")

# Clustering Algorithms 
include("clustering.jl")

# Construction Transition Rate Matrices 
include("construct_transition_matrices.jl")

# Utilities 
include("utils.jl")

end # module
