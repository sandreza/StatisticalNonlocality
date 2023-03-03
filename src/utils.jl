using SparseArrays
import SparseArrays: dropzeros!
export histogram

"""
dropzeros!(sparseMatrix, drop_criteria)
# Description
- Drops machine zeros in sparse matrix
# Arguments
- `A`: a sparse matrix
- `drop_criteria`: criteria for dropping entries
# return
- nothing
"""
function dropzeros!(A, drop_criteria)
    i, j = findnz(A)
    for loop in 1:length(i)
        if abs(A[i[loop], j[loop]]) < drop_criteria
            A[i[loop], j[loop]] = 0.0
        end
    end
    dropzeros!(A)
end

"""
droprelativezeros!(A; scale=100)
# Description
- Drops machine zeros in sparse matrix
# Arguments
- `A`: a sparse matrix
- `drop_criteria`: criteria for dropping entries
# return
- nothing
# This function calls 
dropzeros!(A, eps(scale * maximum(da)))
"""
function droprelativezeros!(A; scale=100)
    dropzeros!(A, eps(scale * maximum(abs.(A))))
end

"""
function histogram(
    array;
    bins=minimum([100, length(array)]),
    normalization=:uniform,
    custom_range=false
)
# Description 
- Utility function for barplots in GLMakie 
# Arguments 
- array; one dimensional sequence of numbers 
# Keyword Arguments 
- bins; how many buckets to use, always uniform
- normalization; how much to weight each value of array (default 1/length(array))
- custom_range; range for uniform bucket
"""
function histogram(
    array;
    bins=minimum([100, length(array)]),
    normalization=:uniform,
    custom_range=false
)
    tmp = zeros(bins)
    if custom_range isa Tuple
        down, up = custom_range
    else
        down, up = extrema(array)
    end
    down, up = down == up ? (down - 1, up + 1) : (down, up) # edge case
    bucket = collect(range(down, up, length=bins + 1))
    if normalization == :uniform
        normalization = ones(length(array)) ./ length(array)
    end
    for i in eachindex(array)
        # normalize then multiply by bins
        val = (array[i] - down) / (up - down) * bins
        ind = ceil(Int, val)
        # handle edge cases
        ind = maximum([ind, 1])
        ind = minimum([ind, bins])
        tmp[ind] += normalization[i]
    end
    return (bucket[2:end] + bucket[1:(end-1)]) .* 0.5, tmp
end