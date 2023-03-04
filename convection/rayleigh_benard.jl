using Oceananigans, Statistics, FFTW

@info "Creating Grid"
grid = RectilinearGrid(size=(64,64), x = (0, 2), z = (0, 1), topology=(Bounded, Flat, Bounded))

@info "Setting Boundary Conditions"
b_bcs = FieldBoundaryConditions(bottom=ValueBoundaryCondition(1.0), top=ValueBoundaryCondition(0.0))
u_bcs = FieldBoundaryConditions(bottom=ValueBoundaryCondition(0.0), top=ValueBoundaryCondition(0.0))

Raᶜ = 27π^4 / 4
Ra = 10^8 # 10^10 * 10^(-2) * Raᶜ
Pr = 1.0
ν = (Pr / Ra)^(1/2)
κ = (Pr * Ra)^(-1/2)
@info "constructing hydrostatic model"
model = NonhydrostaticModel(; 
    grid, 
    timestepper = :RungeKutta3, 
    advection = WENO(), 
    closure = ScalarDiffusivity(ν=ν, κ=κ), 
    buoyancy=Buoyancy(model=BuoyancyTracer()),
    tracers = :b,
    boundary_conditions=(; b = b_bcs)
)

u, v, w = model.velocities

uᵢ = rand(size(u)...)
vᵢ = rand(size(v)...)
uᵢ .-= mean(uᵢ)
vᵢ .-= mean(vᵢ)
bᵢ(x,y,z) = 1-z 

set!(model, u=uᵢ, v=vᵢ, b=bᵢ)

simulation = Simulation(model, Δt=0.01, stop_time=10)

progress(sim) = @info string("Iteration: ", iteration(sim), ", time: ", time(sim), ", temperature mean ", mean(model.tracers.b), "temperature variance ", var(model.tracers.b))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))
run!(simulation)
ω = Field(∂x(v) - ∂y(u))
compute!(ω)
vizu = interior(u)
vizω = interior(ω)
vizb = interior(model.tracers.b)
##
@info "plotting"
using GLMakie
fig = Figure()
ax11 = Axis(fig[1,1])
ax12 = Axis(fig[1,2])
heatmap!(ax11, vizω[:,1,:], interpolate = true, colormap = :balance)
heatmap!(ax12, vizb[:,1,:], interpolate = true, colormap = :thermometer)
display(fig)
println(mean(vizb[:, 1, 1]))
println(mean(vizb[:, 1, 64]))