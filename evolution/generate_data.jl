

if isfile(pwd() * "/data/3_state_ou.hdf5.hdf5")
    @info "3_state_ou.hdf5 data already exists. skipping data generation"
else
    include("run_ou_case.jl")
end

if isfile(pwd() * "/data/channel.hdf5")
    @info "channel.hdf5 data already exists. skipping data generation"
else
    include("run_channel_case.jl")
end

if isfile(pwd * "/data/comparison.hdf5")
    @info "ou_comparison data already exists"
else
    include("n_state_ou.jl")
    include("stochastic_advection.jl")
end
