function coarse_grain_operator(identified_states)
    operator = zeros(length(identified_states), sum(length.(identified_states)))
    for (i, states) in enumerate(identified_states)
        for state in states
            operator[i, state] = 1
        end
    end
    return operator
end

# identified_states = [[1,2,3,4,5], [6, 7, 8, 9, 10]]
identified_states = [[8, 1, 4], [10, 2, 6], [3, 5], [7, 9]]
function partition_coordinate(u)
    return u[1]
end
# psort = sortperm(norm.(snapshots))
qnsap = quantile(partition_coordinate.(snapshots))
qsnap = [-Inf, qnsap..., Inf]
identified_states = []
for i in 2:length(qsnap) 
    tmploop = Int64[]
    for (s,snapshot) in enumerate(snapshots)
        if qsnap[i-1] <= partition_coordinate(snapshot) < qsnap[i]
            push!(tmploop, s)
        end
    end
    if length(tmploop) > 0
        push!(identified_states, tmploop)
    end
end
tmp = []
for states in identified_states
    tmp = [states..., tmp...]
end
check = collect(1:length(snapshots))
sdif = setdiff(check, union(tmp))
fixed_up = copy(identified_states)
for i in sdif
    push!(fixed_up, [i])
end

P = coarse_grain_operator(fixed_up)
P⁺ = (P ./ sum(P, dims=2))' # Moore-Penrose pseudoinverse, pinv(P) also works
# reduced transition matrix
Q̂ = P * Q * P⁺

##
coarse_state = Int64[]
for state in current_state
    for i in eachindex(fixed_up)
        if state in fixed_up[i]
            push!(coarse_state, i)
        end
    end
end

Qc = transition_rate_matrix(coarse_state, length(fixed_up); γ=1);

norm(Q̂ - Qc) / norm(Q̂) * 100