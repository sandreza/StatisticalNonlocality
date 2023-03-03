struct DynamicCluster{D, S, T}
    distance::D
    states::S
    threshold::T
end

function update!(dc::DynamicCluster, state)
    distances = [dc.distance(state, s) for s in dc.states]
    if distances .>= dc.threshold
        push!(dc.states, state)
    end
end

function check(dc::DynamicCluster, state)
    distances = [dc.distance(state, s) for s in dc.states]
    if distances .>= dc.threshold
        return true
    end
    return false
end

function assign(dc::DynamicCluster, state)
    distances = [dc.distance(state, s) for s in dc.states]
    return argmin(distances)
end

dc = DynamicCluster(distance, [], 0.1)

# memo to self 
#= 
1. Sparse Arrays for recording operator
2. Something to dynamically construct the operator
3. Different criteria for determining states "an assign function", given a state assign a number
4. Choice of "reaction coordinates" and distances in reaction coordinates 
5. Add a cap to the number of states so that when the cap is hit all markov states will be assocaited with previously scene states 
6. Add dynamic update to states so that one can include the average state within a given cell. i.e. compute both uₘ = ∫Ωₘ and vₘ = ∫ΩₘZ
7. Learn the function that maps markov states to transition rates 
8. Test parametric dependence on things like a control parameter
=#

##
# 1. assign 
# 2. markov state
# 3. Markov Model 

struct MarkovModel{A, M, T}
    assignment::A 
    state::M
    transition::T 
end
#=
struct Assignment{S}

end
=#
mm(u) = mm.assignment(u)

