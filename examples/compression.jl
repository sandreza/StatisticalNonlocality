bigmat = zeros(length(states[1]), length(states))
lines(log.(b) .- log(b[1]))
[bigmat[:, i] = state[:, i] for i in 1:length(states)]
a,b,c = svd(bigmat)
minrez = sum((log.(b) .- log(b[1]) ) .> -8)
a[:,1:minrez] * Diagonal(b[1:minrez]) * c[:,1:minrez]' - bigmat

heatmap(bigmat)