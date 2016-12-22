

# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

verbose = False

# Global settings
grid_size = 2
num_nodes = grid_size**2
num_vehicles = 4
num_passengers = num_vehicles


# Load graph
graph = nx.grid_2d_graph(grid_size, grid_size, periodic=False, create_using=None)
graph = nx.watts_strogatz_graph(num_nodes,3,0.1)

print "Number of nodes in graph: " , nx.number_of_nodes(graph)
print nx.info(graph)

print "Nodes: ", graph.nodes()
print "Edges: ", graph.edges()

#print nx.all_neighbors(graph,0)
#nx.draw(graph)
#plt.show()


# Initialization
#vehicle_pos_init = np.array((num_nodes))
vehicle_node_init = np.random.choice(np.arange(num_nodes), size=num_vehicles, replace=False)
passenger_node_init = np.random.choice(np.arange(num_nodes), size=num_passengers, replace=False)

print "Vehicles, init: ", vehicle_node_init
print "Passengers:     ", passenger_node_init


sp = nx.shortest_path(graph, source=graph.nodes()[0], target=graph.nodes()[1])
print sp

# Compute cost matrix (vehicle to passenger pairings)
allocation_cost = np.zeros((num_vehicles, num_passengers))
for i in range(num_vehicles):
	for j in range(num_passengers):
		# Compute cost of shortest path for all possible allocations
		sp = nx.shortest_path(graph, source=graph.nodes()[vehicle_node_init[i]], target=graph.nodes()[passenger_node_init[j]])
		#allocation_cost[i, j] = sp


# Find optimal allocation (Munkres algorithm)
