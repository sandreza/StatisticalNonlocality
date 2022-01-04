using SparseArrays
import SparseArrays: dropzeros!

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
function droprelativezeros!(A; scale = 100)
    dropzeros!(A, eps(scale * maximum(abs.(A))))
end