function construct_states(state, snapshots, distance; distance_threshold = nothing)
    states = []
    current_state = Int[]
    push!(states, state[:, 1])
    push!(state_counts, 1)
    push!(current_state, 1)
    
    if isnothing(distance_threshold)
        minimal_state_temporal_distance = maximum([distance(state[:, i], state[:, i+1]) for i in 1:snapshots-1])
        distance_threshold = minimal_state_temporal_distance
    end

    for i in 2:snapshots
        candidate_state = state[:, i]
        distances = [distance(candidate_state, s) for s in states]
        if all(distances .> distance_threshold)
            push!(states, candidate_state)
            push!(current_state, length(states))
        else
            push!(current_state, argmin(distances))
        end
    end

    return states, current_state
end

function transition_rate_matrix(markov_chain, number_of_states; γ=1)
    # dynamic container for holding times
    holding_times = [[] for n in 1:number_of_states]
    # dynamic container for states
    state_list = []

    push!(holding_times[markov_chain[1]], γ)
    push!(state_list, markov_chain[1])
 
    M = length(markov_chain)
    # loop through steps
    for i in 2:M
        current_state = markov_chain[i]
        previous_state = markov_chain[i-1]
        if current_state == previous_state
            holding_times[current_state][end] += γ
        else
            push!(state_list, current_state)
            push!(holding_times[current_state], γ)
        end
    end

    # construct transition matrix from state list 
    constructed_T = zeros(number_of_states, number_of_states)

    # first count how often state=i transitions to state=j 
    # the current state corresponds to a column of the matrix 
    # the next state corresponds to the row of the matrix
    number_of_unique_states = length(state_list)
    for i in 1:number_of_unique_states-1
        local current_state = state_list[i]
        local next_state = state_list[i+1]
        constructed_T[next_state, current_state] += 1
    end

    # now we need to normalize
    normalization = sum(constructed_T, dims=1)
    normalized_T = constructed_T ./ normalization

    # now we account for holding times 
    holding_scale = 1 ./ mean.(holding_times)
    for i in 1:number_of_states
        normalized_T[i, i] = -1.0
        normalized_T[:, i] *= holding_scale[i]
    end
    return normalized_T
end

function perron_frobenius_matrix(markov_chain, number_of_states)
    count_matrix = zeros(number_of_states, number_of_states)
    for i in 1:length(markov_chain)-1
        count_matrix[markov_chain[i+1], markov_chain[i]] += 1
    end
    return count_matrix ./ sum(count_matrix, dims=1)
end