# with oceananigans 

using Printf
using Statistics
using JLD2

using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
# using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, solid_node, solid_interface
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, inactive_node, peripheral_node
# solid_node is inactive_node and solid_interface is peripheral_node
using Oceananigans.Operators: Δzᵃᵃᶜ
#using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity

using Random
Random.seed!(1234)

arch = GPU()
with_ridge = false

filename = "eddying_channel_with_beta_test"

# Domain
const Lx = 4000kilometers # zonal domain length [m]
const Ly = 2000kilometers # meridional domain length [m]
const Lz = 3kilometers    # depth [m]

# number of grid points
Nx = 200# 256# 2*128 # default is 100
Ny = 100# 128# 128 # 256 # 256 # default is 100
Nz = 30# 32 # default is 30

save_fields_interval = 50years
stop_time = 10 * 20years + 1day
Δt₀ = 3 * 20minutes # 3*20minutes # default is 15

grid = RectilinearGrid(arch;
    topology=(Periodic, Bounded, Bounded),
    size=(Nx, Ny, Nz),
    halo=(4, 4, 4),
    x=(0, Lx),
    y=(0, Ly),
    z=(-Lz, 0)
)

@info "Built a grid: $grid."

#####
##### Boundary conditions
#####

α = 2e-4     # [K⁻¹] thermal expansion coefficient
g = 9.8061   # [m s⁻²] gravitational constant
cᵖ = 3994.0   # [J K⁻¹] heat capacity
ρ = 1024.0   # [kg m⁻³] reference density

# Different bottom drags depending on with or without ridge 
# accounting for form drag
if with_ridge
    μ = 2e-3
else
    μ = 1e-2
end

parameters = (Ly=Ly,
    Lz=Lz,
    Qᵇ=10 / (ρ * cᵖ) * α * g,         # buoyancy flux magnitude [m² s⁻³]
    y_shutoff=5 / 6 * Ly,             # shutoff location for buoyancy flux [m]
    τ=0.15 / ρ,                       # surface kinematic wind stress [m² s⁻²]
    μ=μ,                            # quadratic bottom drag coefficient []
    ΔB=8 * α * g,                     # surface vertical buoyancy gradient [s⁻²]
    H=Lz,                             # domain depth [m]
    h=1000.0,                         # exponential decay scale of stable stratification [m]
    y_sponge=19 / 20 * Ly,            # southern boundary of sponge layer [m]
    λt=7days,                         # relaxation time scale for the northen sponge [s]
    λs=4e-5, # 3.858024691358025e-6 # 2e-4,                          # relaxation time scale for the surface [s]
    Δt=0.5 * Δt₀, # timestep size, used for sponge
)

@inline relaxation_profile(y, p) = p.ΔB * (y / p.Ly)
@inline function buoyancy_flux(i, j, grid, clock, model_fields, p)
    y = ynode(Center(), j, grid)
    return @inbounds p.λs * (model_fields.b[i, j, grid.Nz] - relaxation_profile(y, p))
end

buoyancy_flux_bc = FluxBoundaryCondition(buoyancy_flux, discrete_form=true, parameters=parameters)

@inline function u_stress(i, j, grid, clock, model_fields, p)
    y = ynode(Center(), j, grid)
    return -p.τ * sin(π * y / p.Ly)
end

u_stress_bc = FluxBoundaryCondition(u_stress, discrete_form=true, parameters=parameters)

if with_ridge
    @inline is_immersed_drag_u(i, j, k, grid) = Int(peripheral_node(Face(), Center(), Center(), i, j, k - 1, grid) & !inactive_node(Face(), Center(), Center(), i, j, k, grid))
    @inline is_immersed_drag_v(i, j, k, grid) = Int(peripheral_node(Center(), Face(), Center(), i, j, k - 1, grid) & !inactive_node(Center(), Face(), Center(), i, j, k, grid))

    # Keep a constant linear drag parameter independent on vertical level
    @inline u_drag(i, j, k, grid, clock, fields, μ) = @inbounds -μ * sqrt(model_fields.u[i, j, k]^2 + model_fields.v[i, j, k]^2) * is_immersed_drag_u(i, j, k, grid) * fields.u[i, j, k] / Δzᵃᵃᶜ(i, j, k, grid)
    @inline v_drag(i, j, k, grid, clock, fields, μ) = @inbounds -μ * sqrt(model_fields.u[i, j, k]^2 + model_fields.v[i, j, k]^2) * is_immersed_drag_v(i, j, k, grid) * fields.v[i, j, k] / Δzᵃᵃᶜ(i, j, k, grid)

    Fu = Forcing(u_drag, discrete_form=true, parameters=parameters)
    Fv = Forcing(v_drag, discrete_form=true, parameters=parameters)

    u_drag_bc = FluxBoundaryCondition(u_drag, discrete_form=true, parameters=parameters)
    v_drag_bc = FluxBoundaryCondition(v_drag, discrete_form=true, parameters=parameters)

    u_bcs = FieldBoundaryConditions(top=u_stress_bc, bottom=u_drag_bc)
    v_bcs = FieldBoundaryConditions(bottom=v_drag_bc)

else
    @inline u_drag(i, j, grid, clock, model_fields, p) = @inbounds -p.μ * model_fields.u[i, j, 1] * sqrt(model_fields.u[i, j, 1]^2 + model_fields.v[i, j, 1]^2)
    @inline v_drag(i, j, grid, clock, model_fields, p) = @inbounds -p.μ * model_fields.v[i, j, 1] * sqrt(model_fields.u[i, j, 1]^2 + model_fields.v[i, j, 1]^2)
    u_drag_bc = FluxBoundaryCondition(u_drag, discrete_form=true, parameters=parameters)
    v_drag_bc = FluxBoundaryCondition(v_drag, discrete_form=true, parameters=parameters)
    u_bcs = FieldBoundaryConditions(top=u_stress_bc, bottom=u_drag_bc)
    v_bcs = FieldBoundaryConditions(bottom=v_drag_bc)
end

b_bcs = FieldBoundaryConditions(top=buoyancy_flux_bc)

#####
##### Coriolis
#####

const f = -1e-4     # [s⁻¹]
const β = 1e-11    # [m⁻¹ s⁻¹]
coriolis = BetaPlane(f₀=f, β=β)

#####
##### Forcing and initial condition
#####
@inline initial_buoyancy(z, p) = p.ΔB * (exp(z / p.h) - exp(-p.Lz / p.h)) / (1 - exp(-p.Lz / p.h))
@inline mask(y, p) = max(0.0, y - p.y_sponge) / (Ly - p.y_sponge)

@inline function buoyancy_sponge(i, j, k, grid, clock, model_fields, p)
    timescale = p.λt
    y = ynode(Center(), j, grid)
    z = znode(Center(), k, grid)
    target_b = initial_buoyancy(z, p)
    b = @inbounds model_fields.b[i, j, k]
    return -1 / timescale * mask(y, p) * (b - target_b)
end

Fb = Forcing(buoyancy_sponge, discrete_form=true, parameters=parameters)

ridge(x, y) = 1.5e3 * exp(-(x - Lx / 2)^2 / (Lx / 20)^2) - 3e3 #

@inline function u_sponge(i, j, k, grid, clock, model_fields, p)
    timescale = p.Δt # 15 minutes
    x = xnode(Face(), i, grid)
    y = ynode(Center(), j, grid)
    z = znode(Center(), k, grid)
    u = @inbounds model_fields.u[i, j, k]
    mm = (ridge(x, y) >= z) # && (y < 5e5) # e.g. if (x = 1e6, y = 0, z = -3000) this returns true
    return -1 / timescale * mm * u
end
@inline function v_sponge(i, j, k, grid, clock, model_fields, p)
    timescale = p.Δt
    x = xnode(Center(), i, grid)
    y = ynode(Face(), j, grid)
    z = znode(Center(), k, grid)
    v = @inbounds model_fields.v[i, j, k]
    mm = (ridge(x, y) >= z) # && (y < 5e5) # e.g. if (x = 1e6, y = 0, z = 0) this returns false
    return -1 / timescale * mm * v
end
@inline function w_sponge(i, j, k, grid, clock, model_fields, p)
    timescale = p.Δt
    x = xnode(Center(), i, grid)
    y = ynode(Center(), j, grid)
    z = znode(Face(), k, grid)
    w = @inbounds model_fields.w[i, j, k]
    mm = (ridge(x, y) >= z) # && (y < 5e5) # e.g. if (x = 1e6, y = 0, z = 0) this returns false
    return -1 / timescale * mm * w
end

Fu_sponge = Forcing(u_sponge, discrete_form=true, parameters=parameters)
Fv_sponge = Forcing(v_sponge, discrete_form=true, parameters=parameters)
Fw_sponge = Forcing(w_sponge, discrete_form=true, parameters=parameters)

# Turbulence closures

κh = 0.5e-5 # [m²/s] horizontal diffusivity
νh = 30.0   # [m²/s] horizontal viscocity
κz = 0.5e-5 # [m²/s] vertical diffusivity
νz = 3e-4   # [m²/s] vertical viscocity

horizontal_diffusive_closure = HorizontalScalarDiffusivity(ν=νh, κ=κh)


#####
##### Model building
#####

@info "Building a model..."

if with_ridge
    immersed_grid = ImmersedBoundaryGrid(grid, GridFittedBottom(ridge))
    grid = immersed_grid
    boundary_conditions = (; b=b_bcs, u=u_bcs)
    forcings = (; b=Fb, u=Fu, v=Fv)
else
    boundary_conditions = (; b=b_bcs, u=u_bcs, v=v_bcs)
    forcings = (; b=Fb, u=Fu_sponge, v=Fv_sponge, w=Fw_sponge)
end

model = NonhydrostaticModel(;
    grid=grid,
    advection=WENO(),
    buoyancy=BuoyancyTracer(),
    coriolis=coriolis,
    closure=(horizontal_diffusive_closure,),
    tracers=(:b,),
    boundary_conditions=(b=b_bcs, u=u_bcs, v=v_bcs),
    forcing=forcings,
    timestepper=:RungeKutta3
)
@info "Built $model."

#####
##### Initial conditions
#####

# resting initial condition
ε(σ) = σ * randn()
bᵢ(x, y, z) = parameters.ΔB * (exp(z / parameters.h) - exp(-Lz / parameters.h)) / (1 - exp(-Lz / parameters.h)) + ε(1e-8)
uᵢ(x, y, z) = ε(1e-8)
vᵢ(x, y, z) = ε(1e-8)
wᵢ(x, y, z) = ε(1e-8)

Δy = 100kilometers
Δz = 100
Δc = 2Δy
cᵢ(x, y, z) = exp(-(y - Ly / 2)^2 / 2Δc^2) * exp(-(z + Lz / 4)^2 / 2Δz^2)

set!(model, b=bᵢ, u=uᵢ, v=vᵢ, w=wᵢ,) #  c=cᵢ)

#####
##### Simulation building
#####

simulation = Simulation(model, Δt=Δt₀, stop_time=stop_time)

# add progress callback
wall_clock = [time_ns()]

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, next Δt: %s\n",
        100 * (sim.model.clock.time / sim.stop_time),
        sim.model.clock.iteration,
        prettytime(sim.model.clock.time),
        prettytime(1e-9 * (time_ns() - wall_clock[1])),
        maximum(abs, sim.model.velocities.u),
        maximum(abs, sim.model.velocities.v),
        maximum(abs, sim.model.velocities.w),
        prettytime(sim.Δt))

    wall_clock[1] = time_ns()

    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(100))

@info "Running the simulation..."
@info "Simulation completed in " * prettytime(simulation.run_wall_time)

##

using LinearAlgebra

bs = []
us = []
vs = []
ws = []
markov_states = []

Nd = floor(Int, 4 * 60 * 86400 / Δt₀)
for j in ProgressBar(1:100)
    for i in 1:Nd
        time_step!(simulation.model, Δt₀)
    end
    push!(us, Array(interior(simulation.model.velocities.u)))
    push!(vs, Array(interior(simulation.model.velocities.v)))
    push!(ws, Array(interior(simulation.model.velocities.w)))
    push!(bs, Array(interior(simulation.model.tracers.b)))
    push!(markov_states, [us[j], vs[j], ws[j], bs[j]])
    # push!(ηs, Array(interior(simulation.model.free_surface.η)))
end

##
function distance(total_state, markov_state)
    u, v, w, b = total_state
    uₘ, vₘ, wₘ, bₘ = markov_state
    normu = norm(uₘ - u)
    normv = norm(vₘ - v)
    normw = norm(wₘ - w)
    normb = norm(bₘ - b)
    return normu + normv + normw + normb / 0.015
end
##
sampling_rate = 0.25 * 86400
total_time = 100 * 365 * 86400
current_state = Int64[]

Nd = floor(Int, sampling_rate / Δt₀)
simtime = floor(Int, total_time / sampling_rate)
for j in ProgressBar(1:simtime)
    for i in 1:Nd
        time_step!(simulation.model, Δt₀)
    end
    current_u = Array(interior(simulation.model.velocities.u))
    current_v = Array(interior(simulation.model.velocities.v))
    current_w = Array(interior(simulation.model.velocities.w))
    current_b = Array(interior(simulation.model.tracers.b))
    total_state = [current_u, current_v, current_w, current_b]
    push!(current_state, argmin([distance(total_state, markov_state) for markov_state in markov_states]))
end

##
stragglers = setdiff(1:maximum(current_state), union(current_state))

current_state = [current_state..., stragglers..., current_state[1]]
count_matrix = zeros(maximum(current_state), maximum(current_state));
for i in 1:length(current_state)-1
    count_matrix[current_state[i+1], current_state[i]] += 1
end

column_sum = sum(count_matrix, dims=1) .> 0;
row_sum = sum(count_matrix, dims=2) .> 0;
if all(column_sum[:] .== row_sum[:])
    reduced_count_matrix = count_matrix[column_sum[:], column_sum[:]]
else
    println("think harder")
end

count_matrix = reduced_count_matrix

perron_frobenius = count_matrix ./ sum(count_matrix, dims=1)
backwards_frobenius = count_matrix' ./ sum(count_matrix', dims=1)
Λ, V = eigen(perron_frobenius)
iV = inv(V)
p = real.(V[:, end] / sum(V[:, end]))
entropy = sum(-p .* log.(p) / log(length(union(current_state))))
println("The timescales are ", -1 / log(abs(Λ[1])), " to ", -1 / log(abs(Λ[end-1])), " timesteps")
println(" which is ")
dt = Nd * Δt₀ / 86400
println("The timescales are ", -dt / log(abs(Λ[1])), " to ", -dt / log(abs(Λ[end-1])), " days")
