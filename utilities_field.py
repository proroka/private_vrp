import numpy as np
import networkx as nx
import scipy.optimize as opt
import scipy.special as spc
import utilities


# Constants
NORTH = 0
SOUTH = 1
EAST = 2
WEST = 3

# Compute force field
def create_probability_field(graph, passenger_node_indeces):
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
        if np.max(forces) > 0:
            forces /= np.sum(forces)
        field[n] = forces
    return field

def max_field_force(field):
    max_force = 0
    for forces in field.itervalues():
        max_force = max(max_force, np.max(forces))
    return max_force


