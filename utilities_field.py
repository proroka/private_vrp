# Import modules
import numpy as np
import networkx as nx
import scipy.optimize as opt
import scipy.special as spc
import matplotlib.pyplot as plt
# Import my modules
import utilities


# Constants
NORTH = 0
SOUTH = 1
EAST = 2
WEST = 3

direction_offset = [np.array([0, 1]),  # north
                    np.array([0, -1]), # south
                    np.array([1, 0]),  # east
                    np.array([-1, 0])] # west

# Compute force field
def create_probability_field(graph, passenger_node_indeces, alpha=1.):
    passenger_nodes = utilities.index_to_location(graph, passenger_node_indeces)
    field = dict()
    for n in graph.nodes():
        # n is a tuple, node is an array
        node = np.array(n)
        # Outgoing forces from this node
        # Convention: North, South, East, West
        forces = np.zeros(4)
        for pnode in passenger_nodes:
            headings = np.zeros(4)
            # Compute shortest path length
            delta = pnode - node
            length = np.sum(np.abs(node - pnode))
            if length == 0:
                length = 0.01
            # Direction to passenger node
            if delta[0] < 0:
                headings[WEST] = 1 
            if delta[0] > 0:
                headings[EAST] = 1 
            if delta[1] < 0:
                headings[SOUTH] = 1 
            if delta[1] > 0:
                headings[NORTH] = 1 

            # Scale direction by importance (function of path length)
            forces += headings * (1./float(length))

        # Scale locally
        ind = np.where(forces>0)
        forces[ind] = np.power(forces[ind], alpha)
        if np.max(forces) > 0:
            forces /= np.sum(forces)
        field[n] = forces
    return field

def max_field_force(field):
    max_force = 0
    for forces in field.itervalues():
        max_force = max(max_force, np.max(forces))
    return max_force


# Plot graph of field with passengers marked
def plot_field(field, max_force, graph, cell_size, passenger_node_ind):
    pos = dict(zip(graph.nodes(), np.array(graph.nodes())*cell_size)) # Only works for grid graph
    #nx.draw_networkx(graph, pos=pos, with_labels=False, node_size=250, node_color='lightblue')
    passenger_nodes = utilities.index_to_location(graph, passenger_node_ind)
    plt.scatter(passenger_nodes[:,0]*cell_size, passenger_nodes[:,1]*cell_size, color='red',s=400)

    s = cell_size / max_force / 2.
    hw = 2
    hl = 4
    for n in graph.nodes():
        node = np.array(n)
        x = node[0] * cell_size
        y = node[1] * cell_size
        # North
        if field[n][NORTH]:
            plt.quiver(x, y, 0, field[n][NORTH] * s, scale=1., units='xy', headwidth=hw, headlength=hl)
        # South
        if field[n][SOUTH]:
            plt.quiver(x, y, 0, -field[n][SOUTH] * s, scale=1., units='xy', headwidth=hw, headlength=hl)
        # East
        if field[n][EAST]:
            plt.quiver(x, y, field[n][EAST] * s, 0, scale=1., units='xy', headwidth=hw, headlength=hl)
        # West
        if field[n][WEST]:
            plt.quiver(x, y, -field[n][WEST] * s, 0, scale=1., units='xy', headwidth=hw, headlength=hl)

    plt.axis('equal')
    plt.show()

def run_vehicle_flow(vehicle_nodes, passenger_nodes, graph, field, alpha):
    steps = 0
    waiting_time = []
    occupied_vehicle = np.zeros(len(vehicle_nodes))
    passenger_nodes_set = set(tuple(row) for row in passenger_nodes)
    # While there are still waiting passengers
    while passenger_nodes_set: 
    #for i in range(10):
        print "Waiting passengers: ", len(passenger_nodes_set)
        steps += 1 # time
        # Each vehicle takes 1 step
        for i, vnode in enumerate(vehicle_nodes):
            # Don't do anything for occupied vehicles
            if occupied_vehicle[i]:
                continue
            pvalues = field[tuple(vnode)]
            heading = np.random.choice(range(4), p=pvalues)
            # Add the offset direction to vehicle pos
            vehicle_nodes[i,:] = vnode + direction_offset[heading]
            vnode_t = tuple(vehicle_nodes[i,:])
            # Check if passenger is picked up
            if vnode_t in passenger_nodes_set:
                print "Vehicle %d occupied" % i
                # Vehicle i is occupied, store pickup time, remove passenger
                occupied_vehicle[i] = True
                passenger_nodes_set.remove(vnode_t)
                waiting_time.append(steps)
                # Update field
                passenger_node_ind = utilities.location_to_index(graph, passenger_nodes_set)
                field = create_probability_field(graph, passenger_node_ind, alpha)
                max_force = max_field_force(field)
                
    return waiting_time






