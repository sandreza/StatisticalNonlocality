function rare_event(u)
    return all(u[1:4] .> 3)
end
# remove rare events from snapshots 
snapshots = snapshots[(!).(rare_event.(snapshots))]
nstates = length(snapshots)
rare_snapshots = []
for i in 1:size(training_set, 2)
    snapshot = training_set[:, i]
    if rare_event(snapshot)
        push!(rare_snapshots, snapshot)
    end
    toc = time()
    if toc - tic > 1
        println("currently at ", i, " out of ", size(training_set, 2))
        println("added ", length(rare_snapshots), " rare events")
        tic = time()
    end
end

##
# Train Differently
current_state = Int64[]
tic = time()
for i in 1:size(training_set, 2)
    snapshot = training_set[:, i]
    if rare_event(snapshot)
        distances = [distance(snapshot, s) for s in rare_snapshots]
        push!(current_state, length(snapshots) + argmin(distances))
    else
        distances = [distance(snapshot, s) for s in snapshots]
        push!(current_state, argmin(distances))
    end

    toc = time()
    if toc - tic > 1
        println("currently at ", i, " out of ", size(training_set, 2))
        tic = time()
    end
end
# combine states
snapshots = vcat(snapshots, rare_snapshots)