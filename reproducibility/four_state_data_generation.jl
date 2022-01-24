include("four_state_function.jl")

parameter_list = []
filename_list = []

# local
parameters = (; U = 1e0, γ = 1e2, κ = 1e-2, ω = 1e2)
filename = "local.jld2"
push!(parameter_list, parameters)
push!(filename_list, filename)
# nonlocal
parameters = (; U = 1e0, γ = 1e0, κ = 1e0, ω = 1e0)
filename = "nonlocal.jld2"
push!(parameter_list, parameters)
push!(filename_list, filename)
# nonlocal symmetric
parameters = (; U = 1e0, γ = 1e0, κ = 1e0, ω = 0.0)
filename = "nonlocal_symmetric.jld2"
push!(parameter_list, parameters)
push!(filename_list, filename)
# nonlocal
parameters = (; U = sqrt(10), γ = 1e0, κ = 1e0, ω = 1e0)
filename = "nonlocal_more_velocity.jld2"
push!(parameter_list, parameters)
push!(filename_list, filename)

total_time = 0
for (parameters, filename) in zip(parameter_list, filename_list)
    println("------")
    println("Currently on ", filename)
    tic = Base.time()
    four_state(parameters; M = 8 * 8, N = 8, filename = filename, dirichlet = false)
    toc = Base.time()
    println("Finished running in ", toc - tic, " seconds.")
    total_time += toc - tic
    println("------")
end
println("Total time to reproduce data = ", total_time, " seconds")
