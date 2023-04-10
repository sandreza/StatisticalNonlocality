using Rasters
using CairoMakie
using Makie.GeometryBasics
using DataInterpolations, Printf
# First, we get the rasters.  This is made really easy with 
# Rasters.jl, and its integration with RasterDataSources.jl.
# This allows you to effortlessly get rasters from several
# popular datasets on demand, like WorldClim and MODIS!
worldclim_stacks = [RasterStack(WorldClim{Climate}, month=i) for i in 1:12]
# You'll notice that these are RasterStacks.  In order to get them into a form which 
# we can use quickly and efficiently with Makie, we can call `Makie.convert_arguments`
# directly on these rasters, which will return a tuple like `(x, y, z)`, where 
# `z` are the values of the raster at the points `(x, y)`.  This is the format
# that we can supply directly to Makie.
temp_rasters = getproperty.(worldclim_stacks, :tmax)
prec_rasters = getproperty.(worldclim_stacks, :prec)
temp_interpolated = DataInterpolations.QuadraticInterpolation(temp_rasters, 1:length(temp_rasters))
prec_interpolated = DataInterpolations.QuadraticInterpolation(prec_rasters, 1:length(temp_rasters))
# Let's see if this interpolation worked.  We plot an animation:
fig, ax, plt = surface(temp_interpolated(1.0); axis=(; type=Axis3))
hm = heatmap!(ax, temp_interpolated(1.0); nan_color=:black)
translate!(hm, 0, 0, -30)
@time record(fig, "temperature_surface_animation.mp4", LinRange(1, 12, 120); framerate=30) do i
    ax.title[] = @sprintf "%.2f" i
    plt.input_args[1][] = temp_interpolated(i)
    hm.input_args[1][] = temp_interpolated(i)
end
# We want to represent the Earth as a sphere, 
# so we tesselate (break up in to triangles) a sphere, 
# which represents Earth, into a mesh.
m = Makie.GeometryBasics.uv_normal_mesh(
    Makie.GeometryBasics.Tesselation(
        Sphere(
            Point3f(0), 1.0f0
        ),
        200
    )
)
# points = GeometryBasics.coordinates(m)
# GeometryBasics.pointmeta(m, color=rand(length(points)))
# GeometryBasics.pop_pointmeta(m, :uv)
p = decompose(Point3f0, m)
uv = decompose_uv(m)
norms = decompose_normals(m)
# Now, we move to the visualization!
# Let's first define a colormap which we'll use to plot:
cmap = [:darkblue, :deepskyblue2, :deepskyblue, :gold, :tomato3, :red, :darkred]
# We create the Figure, which is the top-level object in Makie,
# and holds the axis which holds our plots.
fig = Figure()
# Now, we plot the sphere, which displays temperature.
ax, temperature_plot = mesh(
    fig[1, 1],
    m;
    color=Makie.convert_arguments(Makie.ContinuousSurface(), worldclim_stacks[10].tmax)[3]',
    colorrange=(-50, 50),
    colormap=cmap,
    shading=true,
    axis=(type=LScene, show_axis=false)
)
fig
# Next, we plot the water as a meshscatter plot, in this case
# kind of like a 3-D barplot on the sphere.
function watermap(uv, water, normalization=908.0f0 * 4.0f0)
    markersize = map(uv) do uv
        wsize = reverse(size(water))
        wh = wsize .- 1
        x, y = round.(Int, Tuple(uv) .* wh) .+ 1
        val = water[size(water)[1]-(y-1), x] / normalization
        (isnan(val) || (val < 0.0)) ? -1.0f0 : val
    end
end
# We don't want to call `convert_arguments` all the time, so let's 
# define a function to do it:
raster2array(raster) = Makie.convert_arguments(Makie.ContinuousSurface(), raster)[3]
watervals = watermap(uv, raster2array(worldclim_stacks[1].prec))
xy_width = 0.01
prec_plot = meshscatter!(
    ax,
    p, # the positions of the tessellated mesh we got last time
    rotations=norms, # rotate according to the normal vector, pointing out of the sphere
    marker=Rect3(Vec3f(0), Vec3f(1)), # unit box
    markersize=Vec3f0.(xy_width, xy_width, watervals), # scale by 0.01 in x and y, and `watervals` in z
    color=watervals,
    colormap=[(:black, 0.0), (:skyblue2, 0.6)],
    shading=false
)
# Before we animate, let's change the camera angle a bit.
eye_position, lookat, upvector = Vec3f0(0.5, 0.8, 2.5), Vec3f0(0), Vec3f0(0, 1, 0)
update_cam!(ax.scene, eye_position, lookat, upvector)
# Now, we animate the water and temperature plots:
@time record(fig, "worldclim_visualization.mp4", LinRange(1, 24, 480); framerate=60) do i
    temperature_plot.color[] = raster2array(temp_interpolated(i % 12))'
    watervals = watermap(uv, raster2array(prec_interpolated(i % 12)))
    prec_plot.color[] = watervals
    prec_plot.markersize[] .= Vec3f0.(xy_width, xy_width, watervals)
    ## since we modify markersize inplace above, we need to notify the signal
    notify(prec_plot.markersize)
end