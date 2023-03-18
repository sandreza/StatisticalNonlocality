import LinearAlgebra: ×
import Base: ^, getindex, ndims

export Circle, S¹, Torus
export ×, ndims, info


struct Circle{ℱ}
    a::ℱ
    b::ℱ
end

function Circle(b::FT) where {FT}
    return Circle(FT(0), b)
end

function Circle()
    return Circle(2π)
end

S¹ = Circle

function Base.show(io::IO, Ω::Circle)
    printstyled(io, "[", color = 226)
    a = @sprintf("%.3f", Ω.a)
    b = @sprintf("%.3f", Ω.b)
    printstyled("$a, $b", color = 7)
    printstyled(io, ")", color = 226)
end

struct Torus{DT}
    domains::DT
end

function Base.show(io::IO, Ω::Torus)
    for (i, domain) in enumerate(Ω.domains)
        print(domain)
        if i != length(Ω.domains)
            printstyled(io, "×", color = 118)
        end
    end
end

function ndims(Ω::Circle)
    return 1
end

function ndims(Ω::Torus)
    return length(Ω.domains)
end

getindex(t::Torus, i) = t.domains[i]

# Algebra
×(arg1::Circle, arg2::Circle) = Torus((arg1, arg2))
×(args::Torus, arg2::Circle) = Torus((args.domains..., arg2))
×(arg1::Circle, args::Torus) = Torus((arg1, args.domains...))
×(arg1::Torus, args::Torus) = Torus((arg1.domains..., args.domains...))
×(args::Torus) = Torus(args...)
# Exponentiation
^(circle::Circle, n::Int) = ^(circle, Val(n))
^(circle::Circle, ::Val{1}) = circle
^(circle::Circle, ::Val{2}) = circle × circle
^(circle::Circle, ::Val{n}) where {n} = circle × ^(circle, Val(n - 1))


function info(Ω::Torus)
    println("This is a ", ndims(Ω), "-dimensional Torus.")
    print("The domain is ")
    println(Ω, ".")
    for (i, domain) in enumerate(Ω.domains)
        domain_string = "periodic"
        length = @sprintf("%.2f ", domain.b - domain.a)
        println("The dimension $i domain is ", domain_string, " with length ≈ ", length)

    end
    return nothing
end

import Base: getindex
import Base: size

export FourierGrid
export create_grid, getindex, size

struct FourierGrid{𝒢,𝒲,𝒟}
    nodes::𝒢
    wavenumbers::𝒲
    domain::𝒟
end


getindex(f::FourierGrid, i) = f.nodes[i]

function FourierGrid(grid_points, Ω::Circle; arraytype = Array) # perhaps change to match the product domain
    @assert length(grid_points) == 1
    grid = arraytype(fourier_nodes(grid_points, a = Ω.a, b = Ω.b))
    k⃗ = arraytype(fourier_wavenumbers(grid_points, L = Ω.b - Ω.a))
    return FourierGrid([grid], [k⃗], Ω)
end

"""
FourierGrid(grid_points, Ω::ProductDomain, arraytype=Array)
# Description
Create a numerical grid with grid_points resolution in the domain Ω \n 
Only works for fully periodic grids at the moment
# Arguments
- `grid_points`: tuple | a tuple of ints in each direction for product domain
- `Ω`: ProductDomain   | a product domain object
# Keyword Arguments
- arraytype = Array
# Return
A Fourier Grid object
"""
function FourierGrid(grid_points::Tuple, Ω::Torus; arraytype = Array)
    @assert length(grid_points) == length(Ω.domains)
    grid = []
    k⃗ = []
    for (i, domain) in enumerate(Ω.domains)
        L = domain.b - domain.a
        reshape_dims = appropriate_dims(length(Ω.domains), i, grid_points[i])
        push!(grid, arraytype(reshape(fourier_nodes(grid_points[i], a = domain.a, b = domain.b), reshape_dims)))
        push!(k⃗, arraytype(reshape(fourier_wavenumbers(grid_points[i], L = L), reshape_dims)))
    end
    return FourierGrid(Tuple(grid), Tuple(k⃗), Ω)
end

FourierGrid(grid_points::Int, Ω::Torus; arraytype = Array) = FourierGrid(Tuple(fill(grid_points, ndims(Ω))), Ω, arraytype = arraytype)
FourierGrid(grid_points::Array, Ω::Torus; arraytype = Array) = FourierGrid(Tuple(grid_points), Ω, arraytype = arraytype)

function Base.show(io::IO, F::FourierGrid)
    println("domain:", F.domain)
    print("gridpoints:")
    if typeof(F.nodes[1]) <: AbstractArray
        for (i, grid) in enumerate(F.nodes)
            print(length(grid))
            if i != length(F.nodes)
                printstyled(io, "×")
            end
        end
    else
        print(length(F.nodes))
    end
end

size(f::FourierGrid) = length.(f.nodes)


"""
appropriate_dims(n₁, n₂, N)
# Description
Create an array of size n₁, with the value N at the location n₂, and ones elswhere
# Arguments
- `n₁`: int | size of the array
- `n₂`: int | location of modification
- `N` : int | value at the modification index n2
# Return
A tuple with all 1's except at location n₂, where it is N
"""
function appropriate_dims(n₁, n₂, N)
    y = ones(Int, n₁)
    y[n₂] = N
    return Tuple(y)
end