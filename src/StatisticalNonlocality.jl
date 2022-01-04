module StatisticalNonlocality

# Discrete OU process ALA Charlie Doering
include("discrete_ornstein_uhlenbeck.jl")

# Fourier-Chebyshev ALA John Boyd 
include("spectral.jl")

# Utilities 
include("utils.jl")

end # module
