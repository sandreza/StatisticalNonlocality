using Enzyme, Random, GLMakie, ProgressBars, Statistics, LinearAlgebra

function henon(sⁿ)
    a = 1.4
    b = 0.3
    xⁿ = sⁿ[1]
    yⁿ = sⁿ[2]
    xⁿ⁺¹ = 1 - a * xⁿ * xⁿ + yⁿ
    yⁿ⁺¹ = b * xⁿ
    return [xⁿ⁺¹, yⁿ⁺¹]
end

function henon_fixed_points(a, b)
    x⁺ = ((b - 1) + sqrt((b - 1)^2 + 4 * a)) / 2a
    x⁻ = ((b - 1) - sqrt((b - 1)^2 + 4 * a)) / 2a

    y⁺ = b * x⁺
    y⁻ = b * x⁻
    return ([x⁺, y⁺], [x⁻, y⁻])
end

function henon⁻¹(sⁿ⁺¹)
    a = 1.4
    b = 0.3
    xⁿ⁺¹ = sⁿ⁺¹[1]
    yⁿ⁺¹ = sⁿ⁺¹[2]
    xⁿ = yⁿ⁺¹ / b
    yⁿ = xⁿ⁺¹ - 1 + a * xⁿ * xⁿ
    return [xⁿ, yⁿ]
end

∇henon(s) = jacobian(Forward, henon, s)

sᶠ = henon_fixed_points(1.4, 0.3)[1]
Random.seed!(12345)
sⁿ = sᶠ + 0.01 * randn(2)
s = Vector{Float64}[]
for i in ProgressBar(1:1000000)
    sⁿ⁺¹ = henon(sⁿ)
    push!(s, sⁿ⁺¹)
    sⁿ = sⁿ⁺¹
end
xs = [s1[1] for s1 in s]
ys = [s2[2] for s2 in s]
hist(xs, bins=100)


snapshots = s[10000:10000:end]
xₘ = [s1[1] for s1 in snapshots]
yₘ = [s2[2] for s2 in snapshots]
sL = length(snapshots)
if sL < 1000
    # for visualization later
    D = [norm(s1 - s2) for s1 in snapshots, s2 in snapshots]
    maxD = maximum(D - I)
    minD = minimum(D + maxD * I)
    𝒦 = @. exp(-5e4 * D^2 / maxD^2)
    function gpr(x, predictor, maxD, snapshots)
        prediction = 0.0
        for i in eachindex(predictor)
            d = norm(x - snapshots[i])
            prediction += predictor[i] * exp(-5e4 * d^2 / maxD^2)
        end
        return prediction
    end
end

current_state = Int64[]
for state in ProgressBar(s)
    push!(current_state, argmin([norm(state - snapshot) for snapshot in snapshots]))
end

count_matrix = zeros(length(snapshots), length(snapshots));
for i in 1:length(current_state)-1
    count_matrix[current_state[i+1], current_state[i]] += 1
end
perron_frobenius = count_matrix ./ sum(count_matrix, dims=1)
Λ, V = eigen(perron_frobenius)
iV = inv(V)
p = real.(V[:, end] / sum(V[:, end]))
entropy = sum(-p .* log.(p) / log(length(snapshots)))

##
mean(xs)
sum(xₘ .* p)
# sum(xₘ .* p) is better than mean(xₘ)
mean(ys)
sum(yₘ .* p)
var(xs)
sum(xₘ .^ 2 .* p) - sum(xₘ .* p)^2
# sum(xₘ .^ 2 .* p) - sum(xₘ .* p)^2 is better than var(xₘ)
var(ys)
sum(yₘ .^ 2 .* p) - sum(yₘ .* p)^2
##
predictor = 𝒦 \ p # visualize probability density p
xvals = range(extrema(xs)..., length=100)
yvals = range(extrema(ys)..., length=100)
p[2] ≈ gpr(snapshots[2], predictor, maxD, snapshots) # check correctness
gpr_prediction = [gpr([x, y], predictor, maxD, snapshots) for x in xvals, y in yvals]
heatmap(gpr_prediction, colorrange=(0, median(p)), colormap=:bone, interpolate=true)
##
fig = Figure()
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])

field = real.(V[:, 1])
predictor = 𝒦 \ (field) # visualize second largest eigenvector
ufield = quantile(abs.(field), 0.5)
colorrange = (-ufield, ufield)
gpr_field1 = [gpr([x, y], predictor, maxD, snapshots) for x in xvals, y in yvals]
heatmap!(ax1, gpr_field1, colorrange=colorrange, colormap=:balance, interpolate=true)

field = imag.(V[:, 1])
predictor = 𝒦 \ (field) # visualize second largest eigenvector
ufield = quantile(abs.(field), 0.5)
colorrange = (-ufield, ufield)
gpr_field2 = [gpr([x, y], predictor, maxD, snapshots) for x in xvals, y in yvals]
heatmap!(ax2, gpr_field2, colorrange=colorrange, colormap=:balance, interpolate=true)
display(fig)

##
# first largest eigenvector without an imaginary component, but still oscillatory  (probably)
index1 = argmax(imag.(Λ) .== 0.0)
# first largest eigenvector without an imaginary component, but not oscillatory (hopefully)
index2 = length(Λ) - argmax(reverse(imag.(Λ) .== 0.0)[2:end])

fig = Figure()
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])

field = real.(V[:, index1])
predictor = 𝒦 \ (field) # visualize second largest eigenvector
ufield = quantile(abs.(field), 0.5)
colorrange = (-ufield, ufield)
gpr_field1 = [gpr([x, y], predictor, maxD, snapshots) for x in xvals, y in yvals]
heatmap!(ax1, gpr_field1, colorrange=colorrange, colormap=:balance, interpolate=true)

field = real.(V[:, index2])
predictor = 𝒦 \ (field) # visualize second largest eigenvector
ufield = quantile(abs.(field), 0.5)
colorrange = (-ufield, ufield)
gpr_field2 = [gpr([x, y], predictor, maxD, snapshots) for x in xvals, y in yvals]
heatmap!(ax2, gpr_field2, colorrange=colorrange, colormap=:balance, interpolate=true)
display(fig)

##
rowsum = sum(perron_frobenius, dims=2)[:]
colprob = [maximum(perron_frobenius[:, i]) for i in 1:sL]
fig = Figure()
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])

field = rowsum
predictor = 𝒦 \ (field) # visualize second largest eigenvector
ufield = quantile(abs.(field), 0.5)
colorrange = (0, ufield)
gpr_field1 = [gpr([x, y], predictor, maxD, snapshots) for x in xvals, y in yvals]
heatmap!(ax1, gpr_field1, colorrange=colorrange, colormap=:afmhot, interpolate=true)

field = colprob
predictor = 𝒦 \ (field) # visualize second largest eigenvector
ufield = quantile(abs.(field), 0.5)
colorrange = (0, ufield)
gpr_field2 = [gpr([x, y], predictor, maxD, snapshots) for x in xvals, y in yvals]
heatmap!(ax2, gpr_field2, colorrange=colorrange, colormap=:afmhot, interpolate=true)
display(fig)