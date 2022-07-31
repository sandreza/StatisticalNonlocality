using StatisticalNonlocality, Distributions, Random, LinearAlgebra
import StatisticalNonlocality: ou_transition_matrix
import StatisticalNonlocality: uniform_phase
import Distributions: Uniform

# Set random seed for reproducibility 
Random.seed!(10001)

# number_of_states
number_of_states = 8
# simulate a continuous time markov process
T = ou_transition_matrix(number_of_states - 1)
# T = uniform_phase(4)

γ = 0.1 # γ is basically the dt of the system let's say
eT = exp(γ * T) # get the transition probabilities
# each column gives the transition probability
# column i, means, given that I am in state i, each row j gives the probability to transition to state j

ceT = cumsum(eT, dims=1)

# Define the jump map
function next_state(current_state_index::Int, cT)
    vcT = view(cT, :, current_state_index)
    u = rand(Uniform(0, 1))
    # choose a random uniform variable and decide next state
    # depending on where one lies on the line with respect to the probability 
    # of being found between given probabilities
    for i in eachindex(vcT)
        if u < vcT[i]
            return i
        end
    end

    return i
end

M = 1000000
markov_chain = zeros(Int, M)
markov_chain[1] = 4
for i = 2:M
    markov_chain[i] = next_state(markov_chain[i-1], ceT)
end

# construct transition probability from markov chain 
constructed_transition_probability = zeros(number_of_states, number_of_states)
for i in 1:M-1
    local current_state = markov_chain[i]
    local next_state = markov_chain[i+1]
    constructed_transition_probability[next_state, current_state] += 1
end

normalization = sum(constructed_transition_probability, dims=1)
constructed_transition_probability = constructed_transition_probability ./ normalization

# dynamic container for holding times
holding_times = [[] for n in 1:number_of_states] 
# dynamic container for states
state_list = []

# initialize
push!(holding_times[markov_chain[1]], γ)
push!(state_list, markov_chain[1])

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
empirical_T = normalized_T

# illustrative example of fooling oneself
# also shows the danger of considering ensemble versus time-averages
count_matrix = zeros(number_of_states, number_of_states)
skip = floor(Int, 6 / γ)
reduced_state_list = copy(markov_chain[1:skip:end])
for i in 1:length(reduced_state_list)-1
    count_matrix[reduced_state_list[i+1], reduced_state_list[i]] += 1
end
lowrez_transition_probability = count_matrix ./ sum(count_matrix, dims=1)

empirical_transition_probability = exp(empirical_T * γ)
empirical_T_2 = log(constructed_transition_probability) / γ
exact_transition_probability = exp(T * γ)
empirical_error = norm(empirical_transition_probability - exact_transition_probability) / norm(exact_transition_probability)
empirical_error_2 = norm(constructed_transition_probability - exact_transition_probability) / norm(exact_transition_probability)
empirical_error_T = norm(empirical_T - T) / norm(T)
empirical_error_2_T = norm(empirical_T_2 - T) / norm(T)
empirical_holding_scale_error = [abs(T[i, i] + holding_scale[i]) / abs(T[i, i]) for i in 1:number_of_states]
empirical_holding_scale_error_2 = [abs(T[i, i] - empirical_T_2[i, i]) / abs(T[i, i]) for i in 1:number_of_states]

println("For the default way of constructing the transition probability")
println("The error in constructed the matrix is ", empirical_error)
println("The relative error in holding scale is ", empirical_holding_scale_error)
println("note, cannot estimate holding scale to within an error less than ", γ, "ish", " for the first case")
println("For the second way of constructing the transition probability")
println("The error in constructed the matrix is ", empirical_error_2)
println("The relative error in holding scale is ", empirical_holding_scale_error_2)
println("This is independent of the choice of timestep, but may yield wonky transition matrices after taking the log")

println("Note that if one does not resolve the slowest decorrelation scale in the process then one gets the following eigenvalue ")
println(eigvals(lowrez_transition_probability)[end-1])
println("as opposed to ", eigvals(exact_transition_probability)[end-1]^skip)

