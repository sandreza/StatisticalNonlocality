using GLMakie
using FFTW

FT = Float64
decay_length = 10^(2+2)
end_wavenumber = 10^(1+2)
decay = 1 ./ (1 .+ range(0, end_wavenumber, length = decay_length)[2:end] .^(1.0))
starting = decay[1]
together_1a = [starting..., decay...]
together_1 = FT.([together_1a..., eps(1.0) + decay[1] , reverse(together_1a[2:end])...])
starting = ones(1)
decay_2 = 1 ./ (1 .+ range(0, end_wavenumber, length = decay_length)[2:end] .^(2))
together_2a = [starting..., (decay_2)...]
together_2 = FT.([together_2a..., eps(1.0) + minimum(decay_2), reverse(together_2a[2:end])...])

fig = Figure()
ax1 = Axis(fig[1, 1]; title = "Log-Log Plot of Kernel in Fourier Space")
label1 = "1/(1 + |k|)"
label2 = "1/(1 + |k|²)"
scatter!(ax1, log10.(collect(1:length(together_1a))), log10.(together_1a), color = (:blue, 0.5), label = label1)
scatter!(ax1, log10.(collect(1:length(together_1a))), log10.(together_2a), color = (:orange, 0.5), label = label2)
axislegend(ax1)
ax2 = Axis(fig[2, 1]; title = "Kernel in Real Space, normalized amplitude to one")
f1 = circshift(real.(fft(together_1)), floor(Int, length(together_1)/2))
f2 = circshift(real.(fft(together_2)), floor(Int, length(together_1)/2))
lines!(ax2, f1 / maximum(f1), color = :blue, label = label1, linewidth = 3)
lines!(ax2, f2 / maximum(f2), color = :orange, label = label2, linewidth = 3)
axislegend(ax2)
ax3 = Axis(fig[3, 1]; title = "Log Plot of Kernel in Real Space, normalized amplitude to one")
lines!(ax3, log.(abs.(f1 / maximum(f1))), color = :blue, label = label1, linewidth = 3)
lines!(ax3, log.(abs.(f2 / maximum(f2))), color = :orange, label = label2, linewidth = 3)
ylims!(ax3, (-20, 1))
axislegend(ax3)
display(fig)