include("plotting_utils.jl")

files = ["local", "nonlocal", "nonlocal_symmetric", "nonlocal_less_diffusivity", "nonlocal_more_velocity", "nonlocal_less_transition", "semilocal", "nonlocal_more_diffusivity_less_transition", "nonlocal_reverse"]

total_time = 0
for file in files
    println("------")
    println("currently on file ", file)
    println("-----")
    tic = Base.time()
    fig = file_kernel_plot(file * ".jld2")
    save("data/" * file * "_kernel.eps", fig)
    fig = file_local_diffusivity_plot(file * ".jld2")
    save("data/" * file * "_local_diffusivity.eps", fig)
    toc = Base.time()

    global total_time += toc - tic
end
println("total time to generate plots ", total_time, " seconds ")