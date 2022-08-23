d = 1000
s = Vector{Float64}[]
push!(s, zeros(d))
Δt = 0.01
for i in ProgressBar(1:100000)
    sⁿ = s[i]
    sⁿ⁺¹ = sⁿ - sⁿ * Δt + √(2Δt) * randn(d)
    push!(s, sⁿ⁺¹)
end

s₁ = [s1[1] for s1 in s]  
r = norm.(s) ./ sqrt(d)
r∞ = norm.(s, Inf)
r1 = norm.(s, 1) ./ d
mean(s₁)
var(s₁)
mean(r)
log10(var(r) / mean(r))
-log10(d) - 0.3
mean(r) / var(r) 
d * 2
var(r) / var(r1)
var(r∞)/ var(r)
##
function histogram(
    array;
    bins=minimum([100, length(array)]),
    normalization=:uniform,
    custom_range=false
)
    tmp = zeros(bins)
    if custom_range isa Tuple
        down, up = custom_range
    else
        down, up = extrema(array)
    end
    down, up = down == up ? (down - 1, up + 1) : (down, up) # edge case
    bucket = collect(range(down, up, length=bins + 1))
    if normalization == :uniform
        normalization = ones(length(array)) ./ length(array)
    end
    for i in eachindex(array)
        # normalize then multiply by bins
        val = (array[i] - down) / (up - down) * bins
        ind = ceil(Int, val)
        # handle edge cases
        ind = maximum([ind, 1])
        ind = minimum([ind, bins])
        tmp[ind] += normalization[i]
    end
    return (bucket[2:end] + bucket[1:(end-1)]) .* 0.5, tmp
end
##
xs₁, ys₁ = histogram(s₁, bins=20, custom_range=extrema(s₁))
xr, yr = histogram(r, bins=maximum([20, floor(Int, sqrt(d))]), custom_range=extrema(r[end-10000:end]))
##
fig = Figure()
ax1 = Axis(fig[1,1])
ax2 = Axis(fig[1,2])
ax3 = Axis(fig[2,1])
ax4 = Axis(fig[2,2])
lines!(ax1, s₁[1:10000])
lines!(ax2, r[1:10000])
barplot!(ax3, xs₁, ys₁, color=:red)
barplot!(ax4, xr, yr, color=:blue)
xlims!(ax4, (0, maximum(r)))
display(fig)