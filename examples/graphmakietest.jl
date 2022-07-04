using StatisticalNonlocality, Distributions, Random, LinearAlgebra
import StatisticalNonlocality: ou_transition_matrix
import StatisticalNonlocality: uniform_phase

cmap = :balance # :Blues_9
cmapa = RGBAf.(to_colormap(cmap), 1);
cmap = vcat(cmapa[1:15], fill(RGBAf(0, 0, 0, 0), 10), cmapa[25:end])


fig = Figure()
ax = Axis(fig[1, 1]; title="Transition Probability", titlesize=30)
ax_Q = Axis(fig[1, 2]; title="Transition Rate", titlesize=30)

dt_slider = Slider(fig[2, 1:2], range=0:0.01:2, startvalue=0)
dt = dt_slider.value
Q = ou_transition_matrix(4)
Q = uniform_phase(4)
T = @lift exp(Q * $dt)

g = DiGraph(exp(Q))
g_Q = DiGraph(Q)


edge_color = @lift [RGBAf(cmapa[4].r, cmapa[4].g, cmap[4].b, $T[i]) for i in 1:ne(g)]
edge_width = [4.0 for i in 1:ne(g)]
arrow_size = [30.0 for i in 1:ne(g)]

edge_color_Q = [RGBAf(cmapa[4].r, cmapa[4].g, cmap[4].b, 1.0) for i in 1:ne(g_Q)]
edge_width_Q = [4.0 for i in 1:ne(g_Q)]
arrow_size_Q = [30.0 for i in 1:ne(g_Q)]

obs_string = @lift("Transition Probability at time t = " * string($dt) )
graphplot!(ax, g, edge_color=edge_color, edge_width=edge_width, arrow_size=arrow_size)

graphplot!(ax_Q, g_Q, edge_color=edge_color_Q, edge_width=edge_width_Q, arrow_size=arrow_size_Q)
display(fig)

