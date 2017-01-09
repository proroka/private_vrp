# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.optimize as opt
import scipy.special as spc
# My modules
import utilities
import utilities_field

set_seed = True
save_plots = True


# Global settings
grid_size = 5
cell_size = 100 # meters
num_nodes = grid_size**2
epsilon = .2
vehicle_density = 0.2
num_vehicles = int(num_nodes * vehicle_density)
num_passengers = num_vehicles

alpha = 1.

# Load graph
graph = nx.grid_2d_graph(grid_size, grid_size, periodic=False, create_using=None)

print nx.info(graph)
print "Number of vehicles: ", num_vehicles

# Initialization
if set_seed: 
    np.random.seed(1234)
vehicle_node_ind = np.random.choice(np.arange(num_nodes), size=num_vehicles, replace=False)
passenger_node_ind = np.random.choice(np.arange(num_nodes), size=num_passengers, replace=False)

# Compute field
field = utilities_field.create_probability_field(graph, passenger_node_ind, alpha)
max_force = utilities_field.max_field_force(field)

utilities_field.plot_field(field, max_force, graph, cell_size, passenger_node_ind, save_fig=True, deterministic=False)


# Plot graph of field with passengers marked
# def plot_field(field, max_force, graph, cell_size, passenger_node_ind, save_fig=False):
#     plt.figure(figsize=(6,6),frameon=False)
#     pos = dict(zip(graph.nodes(), np.array(graph.nodes())*cell_size)) # Only works for grid graph
#     #nx.draw_networkx(graph, pos=pos, with_labels=False, node_size=250, node_color='lightblue')
#     passenger_nodes = utilities.index_to_location(graph, passenger_node_ind)
#     plt.scatter(passenger_nodes[:,0]*cell_size, passenger_nodes[:,1]*cell_size, color='red',s=400)

#     s = cell_size / max_force / 2.
#     hw = 2
#     hl = 4
#     for n in graph.nodes():
#         node = np.array(n)
#         x = node[0] * cell_size
#         y = node[1] * cell_size
#         # North
#         if field[n][utilities_field.NORTH]:
#             plt.quiver(x, y, 0, field[n][utilities_field.NORTH] * s, scale=1., units='xy', headwidth=hw, headlength=hl)
#         # South
#         if field[n][utilities_field.SOUTH]:
#             plt.quiver(x, y, 0, -field[n][utilities_field.SOUTH] * s, scale=1., units='xy', headwidth=hw, headlength=hl)
#         # East
#         if field[n][utilities_field.EAST]:
#             plt.quiver(x, y, field[n][utilities_field.EAST] * s, 0, scale=1., units='xy', headwidth=hw, headlength=hl)
#         # West
#         if field[n][utilities_field.WEST]:
#             plt.quiver(x, y, -field[n][utilities_field.WEST] * s, 0, scale=1., units='xy', headwidth=hw, headlength=hl)

#     plt.axis('equal')
#     plt.show()
#     if save_fig:
#         plt.savefig('figures/quiver_map.png', format='png', transparent=True)



# if save_plots:
#     plt.savefig('figures/quiver_map.eps', format='eps')




