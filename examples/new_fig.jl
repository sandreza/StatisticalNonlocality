mfig = Figure()
ax11 = Axis(mfig[1, 1])
ax12 = Axis(mfig[1, 2])
ax13 = Axis(mfig[1, 3])

sl = Slider(mfig[2, 1:3], range=1:length(us), startvalue=3)
tindex = sl.value
colorrange = (-0.5, 0.5)
# field = @lift(bs[$tindex][:, :, end] .- (mean(bs[$tindex][:, :, end])-mean(mb)) - mb)
field11 = @lift(us[$tindex][:, 1:100, end-2])
field12 = @lift(vs[$tindex][:, 1:100, end-2])
field13 = @lift(bs[$tindex][:, 1:100, end-2])

heatmap!(ax11, field11, colormap=:balance, colorrange=colorrange)
heatmap!(ax12, field12, colormap=:balance, colorrange=colorrange)
heatmap!(ax13, field13, colormap=:thermometer, colorrange=(0, 0.015))
display(mfig)