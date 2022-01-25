using StatisticalNonlocality 
import StatisticalNonlocality: ou_transition_matrix

T = ou_transition_matrix(3)

eT = exp(T)
# each column gives the transition probability
# column i, means, given that I am in state i, each row j gives the probability to transition to state j

