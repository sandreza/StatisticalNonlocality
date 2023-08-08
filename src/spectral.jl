export fourier_nodes, fourier_wavenumbers, cheb

using LinearAlgebra, FFTW, Printf

"""
fourier_nodes(n; a = 0, b = 2π)
# Description
- Create a uniform grid of points for periodic functions
# Arguments
- `N`: integer | number of evenly spaced points 
# Keyword Arguments
- `a`: number | starting point of interval [a, b)
- `b`: number | ending point of interval [a, b)
# Return
- `g`: array | an array of points of evenly spaced points from [a, b)
"""
function fourier_nodes(N; a = 0, b = 2π)
    return (b - a) .* collect(0:(N - 1)) / N .+ a
end

"""
fourier_wavenumbers(N; L = 2π)
# Description
- Create wavenumbers associated with the domain of length L
# Arguments
- `N`: integer | number of wavevectors
# Keyword Arguments
- `L`: number | length of interval [a, b), L = b-a
# Return
- `wavenumbers`: array | an array of wavevectors
"""
function fourier_wavenumbers(N; L = 2π)
    up = collect(0:1:N-1)
    down = collect(-N:1:-1)
    indices = up
    indices[div(N, 2)+1:end] = down[div(N, 2)+1:end]
    indices[1] = 0 # edge case
    # indices[div(N, 2)+1] = 0 # edge case
    wavenumbers = 2π / L .* indices
    return wavenumbers
end

"""
# Description
Spectral differentiation matrix and nodes for periodic domains
# Argument
- 'N': number of gridpoints
# Keyword Argument
a and b specify the interval [a, b)
default assumes x ∈ [0, 2π)
# Return
- 'D': Fourier differentiation matrix
- 'x': Fourier grid points
"""
function fourier(N; a = 0, b = 2π)
    if N == 0
        return [0], [0]
    else
        k = fourier_wavenumbers(N, L = b - a)
        x = fourier_nodes(N, a = a, b = b)
        ℱ = fft(I + zeros(N, N), 1)
        ℱ⁻¹ = ifft(I + zeros(N, N), 1)
        D = real.(ℱ⁻¹ * Diagonal(im .* k) * ℱ)
        return D, x
    end
end

"""
# Description
Julia version of Spectral Methods in Matlab
# Argument
- 'N': polynomial order
# Keyword Argument
a and b specify the interval [a, b]
default assumes x ∈ [-1, 1]
# Return
- 'D': Chebyshev differentiation matrix
- 'x': Guass-Lobatto points
"""
function chebyshev(N; a = -1, b = 1)
    if N == 0
        return [0], [(a + b) / 2]
    else
        x = @. cos(pi * (0:N) / N)
        c = [2; ones(N - 1); 2] .* (-1) .^ (0:N)
        dX = x .- x'
        D = (c ./ c') ./ (dX + I)                # off-diagonal entries
        D = D - Diagonal(sum(D', dims = 1)[1:(N + 1)]) # diagonal entries
        return 2 / (b - a) * D, (b - a) * (x .+ 1) / 2 .+ a
    end
end
