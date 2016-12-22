

# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.optimize as opt

verbose = False
plot_on = True


# Global settings
grid_size = 3
num_nodes = grid_size**2
num_vehicles = 4
num_passengers = num_vehicles


# Load graph
graph = nx.grid_2d_graph(grid_size, grid_size, periodic=False, create_using=None)
#graph = nx.watts_strogatz_graph(num_nodes,3,0.1)

print "Number of nodes in graph: " , nx.number_of_nodes(graph)
print nx.info(graph)

print "Nodes: ", graph.nodes()
print "Edges: ", graph.edges()

if plot_on:
	nx.draw(graph)
	plt.show()


# Initialization
#vehicle_pos_init = np.array((num_nodes))
vehicle_node_init = np.random.choice(np.arange(num_nodes), size=num_vehicles, replace=False)
passenger_node_init = np.random.choice(np.arange(num_nodes), size=num_passengers, replace=False)

print "Vehicles, init: ", vehicle_node_init
print "Passengers:     ", passenger_node_init

# Testing only
sp = nx.shortest_path(graph, source=graph.nodes()[0], target=graph.nodes()[7])
spl = nx.shortest_path_length(graph, source=graph.nodes()[0], target=graph.nodes()[7], weight=None)
print sp
print spl


# Compute cost matrix (vehicle to passenger pairings)
allocation_cost = np.zeros((num_vehicles, num_passengers))
for i in range(num_vehicles):
	for j in range(num_passengers):
		# Compute cost of shortest path for all possible allocations
		allocation_cost[i,j] = nx.shortest_path_length(graph, source=graph.nodes()[vehicle_node_init[i]],
															 target=graph.nodes()[passenger_node_init[j]], weight=None)


# Find optimal allocation (Munkres algorithm)

row_ind, col_ind = opt.linear_sum_assignment(allocation_cost)

print "row: ", row_ind
print "col: ", col_ind

cost = allocation_cost[row_ind, col_ind].sum()

print "total cost: ", cost




