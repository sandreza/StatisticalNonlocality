tfig = Figure()
ax11 = Axis(tfig[1, 1]; title="Zonal Velocity", xlabel="latitude", ylabel="longitude")
ax12 = Axis(tfig[1, 2]; title="Meridional Velocity", xlabel="latitude", ylabel="longitude")
ax13 = Axis(tfig[1, 3]; title="Temperature ", xlabel="latitude", ylabel="longitude")

sl = Slider(tfig[2, 1:3], range=eachindex(us_t), startvalue=1)
tindex = sl.value
colorrange = (-0.5, 0.5)

field11 = @lift(us_t[$tindex][:, 10:90, end-1])
field12 = @lift(vs_t[$tindex][:, 10:90, end-1])
field13 = @lift(bs_t[$tindex][:, 10:90, end-1])
heatmap!(ax11, field11, colormap=:balance, colorrange=colorrange, interpolate=true)
heatmap!(ax12, field12, colormap=:balance, colorrange=colorrange, interpolate=true)
heatmap!(ax13, field13, colormap=:thermometer, colorrange=(0, 0.015), interpolate=true)
display(tfig)

##
# Create illustrative snapshots to describe the method
ius = us_t[1:33:100]
ivs = vs_t[1:33:100]
ibs = bs_t[1:33:100] # [bs[1], bs[2], bs[3], bs[end]] # 

# vizit
sfig = Figure()
ax11 = Axis(sfig[1, 1]; title="Snapshot 1", xlabel="longitude", ylabel="latitude")
ax12 = Axis(sfig[1, 2]; title="Snapshot 2", xlabel="longitude", ylabel="latitude")
ax21 = Axis(sfig[2, 1]; title="Snapshot 3 ", xlabel="longitude", ylabel="latitude")
ax22 = Axis(sfig[2, 2]; title="Snapshot 4 ", xlabel="longitude", ylabel="latitude")

field11 = ibs[1][:, 10:90, end-1]
field12 = ibs[2][:, 10:90, end-1]
field21 = ibs[3][:, 10:90, end-1]
field22 = ibs[4][:, 10:90, end-1]
longitude_c = range(0, 40, length=200)
latitude_c = range(-70, -50, length=100)
heatmap!(ax11, longitude_c, latitude_c, field11, colormap=:thermometer, colorrange=(0, 0.015), interpolate=true)
heatmap!(ax12, longitude_c, latitude_c, field12, colormap=:thermometer, colorrange=(0, 0.015), interpolate=true)
heatmap!(ax21, longitude_c, latitude_c, field21, colormap=:thermometer, colorrange=(0, 0.015), interpolate=true)
heatmap!(ax22, longitude_c, latitude_c, field22, colormap=:thermometer, colorrange=(0, 0.015), interpolate=true)
display(sfig)

##
illustrative_state = Int64[]
for i in eachindex(bs_t)
    push!(illustrative_state, argmin([norm(bs_t[i][:, 10:90, end-1] .- ib[:, 10:90, end-1]) for ib in ibs]))
end
##
fig = Figure(resolution = (1400, 500)) 
ax1 = Axis(fig[1,1])
ax2 = Axis(fig[1,2])
ax3 = Axis(fig[1,3]) 
for ax in [ax1, ax2, ax3]
    hidedecorations!(ax)
    hidespines!(ax)
end
iindex = 4
heatmap!(ax1, ius[iindex][:, 10:90, end-1], colormap = :balance, colorrange = (-0.5,0.5), interpolate = true)
heatmap!(ax2, ivs[iindex][:, 10:90, end-1], colormap=:balance, colorrange=(-0.5, 0.5), interpolate=true)
heatmap!(ax3, ibs[iindex][:, 10:90, end-1], colormap=:thermometer, colorrange= (0, 0.015), interpolate=true)

display(fig)


##
embed_fig = Figure()
ax13 = Axis(embed_fig[1, 3]; title="Markov Embedding", xlabel="time [days]", ylabel="snapshot label", titlesize = 30, ylabelsize=30, xlabelsize=30)
sl = Slider(embed_fig[2, 1:3], range=2:length(bs_t), startvalue=2)
tindex = sl.value

times = floor.(Int, collect(0:length(bs_t)-1) .* dt_t)

field11 = @lift(bs_t[$tindex][:, 10:90, end-1])
fieldindex = @lift(argmin([norm($field11 .- ib[:, 10:90, end-1]) for ib in ibs]))
field12 = @lift(ibs[$fieldindex][:, 10:90, end-1])
field13 = @lift([illustrative_state[1:$tindex]..., zeros(2000 - $tindex)...])

ax11_title = @lift("Dynamics at t=" * string(times[$tindex]) * " days")
ax12_title = @lift("Snapshot " * string($fieldindex))
ax11 = Axis(embed_fig[1, 1]; title=ax11_title, ylabel="latitude [ᵒ]", xlabel="longitude [ᵒ]", titlesize = 30, ylabelsize = 30, xlabelsize = 30)
ax12 = Axis(embed_fig[1, 2]; title=ax12_title, ylabel="latitude [ᵒ]", xlabel="longitude [ᵒ]", titlesize = 30, ylabelsize = 30, xlabelsize = 30)



longitude_c = range(0, 40, length=200)
latitude_c = range(-70, -50, length=100)

heatmap!(ax11, longitude_c, latitude_c, field11, colormap=:thermometer, colorrange=(0, 0.015), interpolate=true)
heatmap!(ax12, longitude_c, latitude_c, field12, colormap=:thermometer, colorrange=(0, 0.015), interpolate=true)
scatter!(ax13, times, field13)
ylims!(ax13, (0.5, 4.5))

display(embed_fig)

##
fig = Figure() 
kwargs = (; titlesize = 30, ylabelsize = 30, xlabelsize = 30)
ax1 = Axis(fig[1,1]; title = "Markov Embedding", xlabel = "time [days]", ylabel = "snapshot label", kwargs...)
scatter!(ax1, 0:2:399, illustrative_state[1:200])
display(fig)


##
frames = eachindex(bs_t)

record(embed_fig, "/Users/andresouza/Desktop/" * "meander_temperature.mp4", frames; framerate=60) do frame
    tindex[] = frame
end

##
count_matrix = zeros(maximum(illustrative_state), maximum(illustrative_state));
for i in 1:length(illustrative_state)-1
    count_matrix[illustrative_state[i+1], illustrative_state[i]] += 1
end

perron_frobenius = count_matrix ./ sum(count_matrix, dims=1)
Q = log(perron_frobenius)
##
count_matrix = zeros(maximum(illustrative_state), maximum(illustrative_state));
for i in 1:1000
    count_matrix[illustrative_state[i+1], illustrative_state[i]] += 1
end
perron_frobenius1 = count_matrix ./ sum(count_matrix, dims=1)
count_matrix = zeros(maximum(illustrative_state), maximum(illustrative_state));
for i in 1000:2000-1
    count_matrix[illustrative_state[i+1], illustrative_state[i]] += 1
end
perron_frobenius2 = count_matrix ./ sum(count_matrix, dims=1)

perron_frobenius1 - perron_frobenius2

##
Λ1, V1 = eigen(perron_frobenius1)
Λ2, V2 = eigen(perron_frobenius2)