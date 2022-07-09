using StatisticalNonlocality
using Distributions, Random, LinearAlgebra
using Graphs
using GLMakie, GraphMakie

import StatisticalNonlocality: ou_transition_matrix
import StatisticalNonlocality: uniform_phase

cmap = :balance # :Blues_9
cmapa = RGBAf.(to_colormap(cmap), 1);
cmap = vcat(cmapa[1:15], fill(RGBAf(0, 0, 0, 0), 10), cmapa[25:end])

fig = Figure(resolution=(1506, 1076))
ax = Axis(fig[1, 1]; title="Transition Probability", titlesize=30)
ax_Q = Axis(fig[1, 2]; title="Transition Rate", titlesize=30)

dt_slider = Slider(fig[2, 1:2], range=0:0.01:2, startvalue=0)
dt = dt_slider.value
# Q = ou_transition_matrix(4)
# Q = uniform_phase(4)
T = @lift exp(Q * $dt)

g = DiGraph(exp(Q))
g_Q = DiGraph(Q)

edge_color = @lift [RGBAf(cmapa[4].r, cmapa[4].g, cmap[4].b, $T[i]) for i in 1:ne(g)]
edge_width = [4.0 for i in 1:ne(g)]
arrow_size = [30.0 for i in 1:ne(g)]
node_labels = repr.(1:nv(g))

edge_color_Q = [RGBAf(cmapa[4].r, cmapa[4].g, cmap[4].b, 1.0) for i in 1:ne(g_Q)]
edge_width_Q = [4.0 for i in 1:ne(g_Q)]
arrow_size_Q = [30.0 for i in 1:ne(g_Q)]
node_labels_Q = reverse(repr.(1:nv(g_Q)))
node_size = 20.0

# obs_string = @lift("Transition Probability at time t = " * string($dt) )
p = graphplot!(ax, g, edge_color=edge_color, edge_width=edge_width,
    arrow_size=arrow_size, node_size=node_size,
    nlabels=node_labels, nlabels_textsize=50.0)

offsets = 0.15 * (p[:node_pos][] .- p[:node_pos][][1])
offsets[1] = Point2f(0, 0.3)
p.nlabels_offset[] = offsets
autolimits!(ax)
hidedecorations!(ax)

p_Q = graphplot!(ax_Q, g_Q, edge_color=edge_color_Q, edge_width=edge_width_Q,
    arrow_size=arrow_size_Q, node_size=node_size, 
    nlabels=node_labels_Q, nlabels_textsize=50.0)

offsets = 0.15 * (p_Q[:node_pos][] .- p_Q[:node_pos][][1])
offsets[1] = Point2f(0, 0.3)
p_Q.nlabels_offset[] = offsets
autolimits!(ax_Q)

hidedecorations!(ax_Q)
display(fig)
