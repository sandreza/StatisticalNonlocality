

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
