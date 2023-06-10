using GLMakie
using FFTW

starting = ones(1)
decay = 1 ./ collect(1:1000)
together_1a = [starting..., decay...]
together_1 = [together_1a..., eps(1.0), reverse(together_1a)[2:end]...]
together_2a = [starting..., (decay .^2)...]
together_2 = [together_2a..., eps(1.0), reverse(together_2a)[2:end]...]

fig = Figure()
ax1 = Axis(fig[1, 1]; title = "Log-Log Plot of Kernel in Fourier Space")
scatter!(ax1, log10.(collect(1:length(together_1a))), log10.(together_1a), color = (:blue, 0.5), label = "k⁻¹")
scatter!(ax1, log10.(collect(1:length(together_1a))), log10.(together_2a), color = (:orange, 0.5), label = "k⁻²")
axislegend(ax1)
ax2 = Axis(fig[2, 1]; title = "Log-Log Plot of Kernel in Real Space, normalized amplitude to one")
f1 = circshift(real.(fft(together_1)), floor(Int, length(together_1)/2))
f2 = circshift(real.(fft(together_2)), floor(Int, length(together_1)/2))
lines!(ax2, f1 / maximum(f1), color = :blue, label = "k⁻¹", linewidth = 3)
lines!(ax2, f2 / maximum(f2), color = :orange, label = "k⁻²", linewidth = 3)
axislegend(ax2)
display(fig)