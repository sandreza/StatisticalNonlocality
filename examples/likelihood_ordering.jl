# Find Cyclic Reductions if possible
likelihood_ordering = [argmax(Q[:, i]) for i in 1:length(snapshots)]

mpp(i) = likelihood_ordering[i]

mpp_dynamics_end = []
for i in ProgressBar(1:length(snapshots))
    initial_i = i
    mpp_dynamics = Int64[]
    for i in 1:100*length(snapshots)
        push!(mpp_dynamics, mpp(initial_i))
        initial_i = mpp(initial_i)
    end
    push!(mpp_dynamics_end, union(mpp_dynamics[end-length(snapshots):end]))
end
cyclic_groups = union(mpp_dynamics_end)
cyclic_groups[5]
cyclic_groups[6]
generators = [sort(generator)[1] for generator in cyclic_groups]
generators = union(generators)

mpp_dynamics_end = []
for i in ProgressBar(generators)
    initial_i = i
    mpp_dynamics = Int64[]
    for i in 1:100*length(snapshots)
        push!(mpp_dynamics, mpp(initial_i))
        initial_i = mpp(initial_i)
    end
    push!(mpp_dynamics_end, union(mpp_dynamics[end-length(snapshots):end]))
end

## Reduced Dynamics: 
# sum(length.(cyclic_groups))

