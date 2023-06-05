@info "convolution plot"
Ms = 0:256
Ns = [1, 2, 3, 4, 14]
keffs = Vector{Float64}[]
for N in [15]
    keff = n_state_keff(N; Ms=Ms)
    push!(keffs, keff)
end

using FFTW
##
N = 256
bump(x; λ=80 / N, width=1) = 0.5 * (tanh((x + width / 2) / λ) - tanh((x - width / 2) / λ))

𝒦̂ = zeros(N)
Mval = 127
𝒦̂[1:1+Mval] = keffs[1][1:1 + Mval]
𝒦̂[end-Mval+1:end] .= reverse(keffs[1][2:1+Mval])
# scatter(x, circshift(real.(fft(𝒦̂)), 129))
# scatter!(x, bump.(x .-π))
Ω = S¹(2π) 
ArrayType = Array
Ns = N
grid = FourierGrid(Ns, Ω, arraytype=ArrayType)
nodes, wavenumbers = grid.nodes, grid.wavenumbers
# build operators
x = nodes[1]
kˣ = wavenumbers[1]
∂x = im * kˣ
Δ = @. ∂x^2
κ = 0.01
𝒟 = κ .* Δ
𝒪 = (𝒦̂ .+ κ) .* Δ 
source_primal = bump.(x .-π)
s = real.(ifft(𝒟 .* fft(source_primal)))
sol = fft(s) ./ 𝒪
sol[isnan.(sol)] .= 0.0
θ̄ = real.(ifft(sol))
∂ˣθ̄ = @.  exp(-2(x - π)^2) - exp(-10(x - π - π/8)^2) # real.(ifft(∂x .* sol))
uθ = -real.(ifft(𝒦̂ .* fft(∂ˣθ̄)))
##
fig = Figure(resolution = (2000, 1000))
options = (; xlabelsize=40, ylabelsize=40, xticklabelsize=30, yticklabelsize=30)
fig = Figure(resolution=(2000, 1000))
ax11 = Axis(fig[1, 1]; options..., xlabel=L"x", ylabel="Amplitude")
ax12 = Axis(fig[1, 2]; options..., xlabel=L"\partial_x \theta", ylabel=L"u \theta")
lines!(ax11, x, ∂ˣθ̄, color = :green, linewidth = 3, label = "Gradient")
# lines!(ax11, x, circshift(real.(fft(𝒦̂)), 129) ./ 10, label = "Kernel x 0.1", color = :purple)
lines!(ax11, x, uθ, color = :orange, linewidth = 3, label = "Flux")
ylims!(ax11, -0.5, 1.0)
axislegend(ax11, position=:rt, framecolor=(:grey, 0.5), patchsize=(20, 20), markersize=20, labelsize=30)
positive_indices = (∂ˣθ̄ ./ uθ) .> 0
negative_indices = (∂ˣθ̄ ./ uθ) .< 0
colors = [positive_index ? :red : :blue for positive_index in positive_indices]
lines!(ax12, ∂ˣθ̄, uθ, color=colors, linewidth=3)
xlims!(ax12, -0.7, 1.1)
ylims!(ax12, -0.4, 0.0)
display(fig)
##
save("data/fig3.eps", fig)
@info "done with convolution plot"