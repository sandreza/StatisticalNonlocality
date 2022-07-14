using LinearAlgebra
import StatisticalNonlocality: ou_transition_matrix

n = 5
M6 = ou_transition_matrix(n)
n = 3
M4 = ou_transition_matrix(n)
n = 1
M2 = ou_transition_matrix(n)

Λ4, V4 = eigen(M4)
Λ2, V2 = eigen(M2)
Λ6, V6 = eigen(M6)

function coarse_grain_operator(identified_states)
    operator = zeros(length(identified_states), sum(length.(identified_states)))
    for (i, states) in enumerate(identified_states)
        for state in states
            operator[i, state] = 1
        end
    end
    return operator
end

identified_states = [[1, 2], [3, 4], [ 5, 6]]
tmp = []
for states in identified_states
    tmp = [states..., tmp...]
end
check = collect(1:6)
sdif = setdiff(check, tmp)
fixed_up = copy(identified_states)
for i in sdif
    push!(fixed_up, [i])
end

P = coarse_grain_operator(fixed_up)
Vr = (P * V6)[:, end-length(fixed_up)+1:end]
Qr = Vr * Diagonal(Λ6[end-length(fixed_up) + 1:end]) * inv(Vr)

Q̂ = P * M6 * pinv(P)