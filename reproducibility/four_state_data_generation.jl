include("four_state_function.jl")

parameter_list = []
filename_list = []

# local
parameters = (; U = 1e0, γ = 1e2, κ = 1e-2, ω = 1e2)
filename = "local.jld2"
push!(parameter_list, parameters)
push!(filename_list, filename)
# semilocal
parameters = (; U = 1e0, γ = 1e1, κ = 1e-1, ω = 1e1)
filename = "semilocal.jld2"
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
# nonlocal more U
parameters = (; U = 1e0, γ = 1e-1, κ = 1e-1, ω = 1e-1)
filename = "nonlocal_more_velocity.jld2"
push!(parameter_list, parameters)
push!(filename_list, filename)
# nonlocal less diffusivity
parameters = (; U = 1e0, γ = 1e-0, κ = 1e-1, ω = 1e-0)
filename = "nonlocal_less_diffusivity.jld2"
push!(parameter_list, parameters)
push!(filename_list, filename)
# nonlocal less transition
parameters = (; U = 1e0, γ = 1e-1, κ = 1e-0, ω = 1e-1)
filename = "nonlocal_less_transition.jld2"
push!(parameter_list, parameters)
push!(filename_list, filename)
# nonlocal more diffisuvity less transition
parameters = (; U = 1e0, γ = 1e-1, κ = 1e1, ω = 1e-1)
filename = "nonlocal_more_diffusivity_less_transition.jld2"
push!(parameter_list, parameters)
push!(filename_list, filename)
# reverse transition
parameters = (; U = 1e0, γ = 1e-0, κ = 1e0, ω = -1e-0)
filename = "nonlocal_reverse.jld2"
push!(parameter_list, parameters)
push!(filename_list, filename)

total_time = 0
for (parameters, filename) in zip(parameter_list, filename_list)
    println("------")
    println("Currently on ", filename)
    tic = Base.time()
    four_state(parameters; M = 8 * 8, N = 16, filename = filename, dirichlet = false)
    toc = Base.time()
    println("Finished running in ", toc - tic, " seconds.")
    global total_time += toc - tic
    println("------")
end
println("Total time to reproduce data = ", total_time, " seconds")
