using JLD2, GLMakie

function kernel_plot(kernel; colormap = :thermal, colormap2 = :balance, resolution = (1800, 1300))
    (; Kˣˣ, Kˣᶻ, Kᶻˣ, Kᶻᶻ) = kernel

    fig = Figure(resolution = resolution)
    titlestring = "Kˣˣ"
    ax1 = Axis(fig[1, 1], title = titlestring, titlesize = 30)
    titlestring = "Kˣᶻ"
    ax2 = Axis(fig[1, 3], title = titlestring, titlesize = 30)
    titlestring = "Kᶻˣ"
    ax3 = Axis(fig[2, 1], title = titlestring, titlesize = 30)
    titlestring = "Kᶻᶻ"
    ax4 = Axis(fig[2, 3], title = titlestring, titlesize = 30)

    hm1 = heatmap!(ax1, Kˣˣ, colormap = colormap)
    ax1.yreversed = true

    hm2 = heatmap!(ax2, Kˣᶻ, colormap = colormap2)
    ax2.yreversed = true

    hm3 = heatmap!(ax3, Kᶻˣ, colormap = colormap2)
    ax3.yreversed = true

    hm4 = heatmap!(ax4, Kᶻᶻ, colormap = colormap)
    ax4.yreversed = true

    Colorbar(fig[1, 2], hm1, height = Relative(3 / 4), width = 25, ticklabelsize = 30,
        labelsize = 30, ticksize = 25, tickalign = 1,)
    Colorbar(fig[2, 2], hm2, height = Relative(3 / 4), width = 25, ticklabelsize = 30,
        labelsize = 30, ticksize = 25, tickalign = 1,)
    Colorbar(fig[1, 4], hm3, height = Relative(3 / 4), width = 25, ticklabelsize = 30,
        labelsize = 30, ticksize = 25, tickalign = 1,)
    Colorbar(fig[2, 4], hm4, height = Relative(3 / 4), width = 25, ticklabelsize = 30,
        labelsize = 30, ticksize = 25, tickalign = 1,)
    display(fig)
    return fig
end

function local_diffusivity_plot(analytic, numerical, z; xlimits = nothing)
    (; a_κˣˣ, a_κˣᶻ, a_κᶻˣ, a_κᶻᶻ) = analytic
    (; κˣˣ, κˣᶻ, κᶻˣ, κᶻᶻ) = numerical

    options = (; ylabel = "z", ylabelsize = 32,
        xlabelsize = 32, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
        xticksize = 30, ytickalign = 1, yticksize = 30,
        xticklabelsize = 30, yticklabelsize = 30)
        
    fig = Figure(resolution = (1800, 1300), title = "Local Operators")
    titlestring = "Kˣˣ"
    ax1 = Axis(fig[1, 1]; options..., title = titlestring, titlesize = 30)
    titlestring = "Kˣᶻ"
    ax2 = Axis(fig[1, 2]; options..., title = titlestring, titlesize = 30)
    titlestring = "Kᶻˣ"
    ax3 = Axis(fig[2, 1]; options..., title = titlestring, titlesize = 30)
    titlestring = "Kᶻᶻ"
    ax4 = Axis(fig[2, 2]; options..., title = titlestring, titlesize = 30)

    plot_string_1 = "analytic"
    plot_string_2 = "numerical"

    ln1 = lines!(ax1, a_κˣˣ, z, color = :red)
    sc1 = scatter!(ax1, κˣˣ, z, color = :blue)
    axislegend(ax1, [ln1, sc1], [plot_string_1, plot_string_2], position = :rc, labelsize = 30)

    ln2 = lines!(ax2, a_κˣᶻ, z, color = :red)
    sc2 = scatter!(ax2, κˣᶻ, z, color = :blue)
    # axislegend(ax2, [ln2, sc2], [plot_string_1, plot_string_2], position = :rt, labelsize = 30)
    hideydecorations!(ax2)

    ln3 = lines!(ax3, a_κᶻˣ, z, color = :red)
    sc3 = scatter!(ax3, κᶻˣ, z, color = :blue)
    # axislegend(ax3, [ln3, sc3], [plot_string_1, plot_string_2], position = :rb, labelsize = 30)

    ln4 = lines!(ax4, a_κᶻᶻ, z, color = :red)
    sc4 = scatter!(ax4, κᶻᶻ, z, color = :blue)
    hideydecorations!(ax4)
    # axislegend(ax4, [ln4, sc4], [plot_string_1, plot_string_2], position = :rt, labelsize = 30)
    # colgap!(ax1, 10)
    # rowgap!(ax1, 10)

    for ax in [ax1, ax2, ax3, ax4]
        if isnothing(xlimits)
        else
            xlims!(ax, xlimits)
        end
    end

    display(fig)
    return fig
end

function file_kernel_plot(file)
    jlfile = jldopen("data/" * file, "a+")
    # jlfile = jldopen("data/nonlocal_more_velocity.jld2")
    EF¹¹ = jlfile["diffusivity"]["K11"]
    EF¹² = jlfile["diffusivity"]["K12"]
    EF²¹ = jlfile["diffusivity"]["K21"]
    EF²² = jlfile["diffusivity"]["K22"]
    x = jlfile["grid"]["x"]
    z = jlfile["grid"]["z"]
    N = length(x)
    M = length(z)
    tmpE = copy(EF¹¹)
    EF¹¹ = reshape(permutedims(reshape(EF¹¹, (N, M, N, M)), (2, 1, 4, 3)), (N * M, N * M))
    EF¹² = reshape(permutedims(reshape(EF¹², (N, M, N, M)), (2, 1, 4, 3)), (N * M, N * M))
    EF²¹ = reshape(permutedims(reshape(EF²¹, (N, M, N, M)), (2, 1, 4, 3)), (N * M, N * M))
    EF²² = reshape(permutedims(reshape(EF²², (N, M, N, M)), (2, 1, 4, 3)), (N * M, N * M))
    kernel_matrix = (; Kˣˣ = EF¹¹, Kˣᶻ = EF¹², Kᶻˣ = EF²¹, Kᶻᶻ = EF²²)
    fig = kernel_plot(kernel_matrix)
    return fig
end

function file_local_diffusivity_plot(file)
    jlfile = jldopen("data/"*file, "a+")

    κ¹¹ = jlfile["averagelocaldiffusivity"]["κ11"]
    κ¹² = jlfile["averagelocaldiffusivity"]["κ12"]
    κ²¹ = jlfile["averagelocaldiffusivity"]["κ21"]
    κ²² = jlfile["averagelocaldiffusivity"]["κ22"]
    z = jlfile["grid"]["z"][:]
    x = jlfile["grid"]["x"][:]
    ##
    # Contruct local diffusivity estimate from flow field and γ
    γ = jlfile["parameters"]["γ"]
    ω = jlfile["parameters"]["ω"]
    u¹ = jlfile["velocities"]["u¹"]
    u² = jlfile["velocities"]["u²"]
    v¹ = jlfile["velocities"]["v¹"]
    v² = jlfile["velocities"]["v²"]

    # integrating out the first dimension amounts to assuming that the dominant mode
    # is the zero'th. 
    N = length(x)
    scale = (2 * (ω^2 + γ^2))
    analytic_κ¹¹ = sum((γ * u¹ .* u¹ + γ * u² .* u²) ./ scale, dims = 1)[:] ./ N
    analytic_κ¹² = sum((u¹ .* (γ * v¹ + ω * v²) + u² .* (γ * v² - ω * v¹)) ./ scale, dims = 1)[:] ./ N
    analytic_κ²¹ = sum((v¹ .* (γ * u¹ + ω * u²) + v² .* (γ * u² - ω * u¹)) ./ scale, dims = 1)[:] ./ N
    analytic_κ²² = sum((γ * v¹ .* v¹ + γ * v² .* v²) ./ scale, dims = 1)[:] ./ N
    analytic = (; a_κˣˣ = analytic_κ¹¹, a_κˣᶻ = analytic_κ¹², a_κᶻˣ = analytic_κ²¹, a_κᶻᶻ = analytic_κ²²)
    numerical = (; κˣˣ = κ¹¹, κˣᶻ = κ¹², κᶻˣ = κ²¹, κᶻᶻ = κ²²)
    fig = local_diffusivity_plot(analytic, numerical, z[:])
    return fig
end
