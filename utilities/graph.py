from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np
import utm
from scipy.spatial import cKDTree


# Normalize the graph nodes to be contiguous indices.
def normalize(graph, route_lengths):
    mapping = {}
    for i, node in enumerate(graph.nodes()):
        mapping[node] = i
    graph = nx.relabel_nodes(graph, mapping=mapping)
    # Update route lengths.
    assert len(mapping) == len(route_lengths)
    new_route_lengths = np.empty((len(mapping), len(mapping)), dtype=np.float32)
    for u, lengths in route_lengths.iteritems():
        assert len(lengths) == len(mapping)
        for v, l in lengths.iteritems():
            new_route_lengths[mapping[u], mapping[v]] = route_lengths[u][v]
    return graph, new_route_lengths, NearestNeighborSearcher(graph)


# Create directional grid map
def create_grid_map(grid_size=10, edge_length=100., default_speed=10., std_dev=2.0):
    nx_graph = nx.grid_2d_graph(grid_size, grid_size)
    graph = nx.MultiDiGraph()
    node_to_index = {}
    for i, (x, y) in enumerate(nx_graph.nodes()):
        graph.add_node(i, x=float(x) * edge_length, y=float(y) * edge_length)
        node_to_index[(x, y)] = i
    for u, v in nx_graph.edges():
        # randomize default speed
        ds = max(1.0, default_speed + np.random.randn()*std_dev)
        graph.add_edge(node_to_index[u], node_to_index[v], length=edge_length,
                     oneway=False, speed=default_speed, time=edge_length / ds)
        graph.add_edge(node_to_index[v], node_to_index[u], length=edge_length,
                     oneway=False, speed=default_speed, time=edge_length / ds)
    return graph

def grid_map_route_lengths(graph):
    route_lengths = dict(nx.shortest_path_length(graph, weight='time'))
    # We need to convert all defaultdict to dict before saving.
    for k, v in route_lengths.iteritems():
        route_lengths[k] = dict((m, n) for m, n in v.iteritems())
    # with open(route_lengths_filename, 'wb') as fp:
    #     fp.write(msgpack.packb(route_lengths))
    
    return route_lengths

class NearestNeighborSearcher(object):

    def __init__(self, graph):
        points = []
        indices = []
        for k, v in graph.node.iteritems():
            points.append([v['x'], v['y']])
            indices.append(k)
        self.indices = np.array(indices)
        self.kdtree = cKDTree(points, 10)

    def Search(self, xy):
        if isinstance(xy, np.ndarray) and xy.shape == 1:
            single_point = True
            xy = [xy]
        else:
            single_point = False
        distances, indices = self.kdtree.query(xy)
        if single_point:
            return self.indices[indices[0]], distances[0]
        return self.indices[indices], distances

    def SearchRadius(self, xy, dist=1.):
        return self.kdtree.query_ball_point(xy, r=dist)

    def SearchK(self, xy, k=1):
        return self.kdtree.query(xy, k=k)


def FromLatLong(lat_long):
    x, y, _, _ = utm.from_latlon(*lat_long)
    return np.array([x, y])


def GetNodePosition(graph, node_id):
    return np.array([graph.node[node_id]['x'], graph.node[node_id]['y']])

def GetNodePositions(graph, node_ids):
    pos = np.zeros((len(node_ids),2))
    for i in range(len(node_ids)):
        pos[i,:] = [graph.node[node_ids[i]]['x'], graph.node[node_ids[i]]['y']]

    return pos


# Modified from https://github.com/gboeing/osmnx/blob/master/osmnx/plot.py.
def PlotRoute(G, route, ax):
    # the origin and destination nodes are the first and last nodes in the route
    origin_node = route[0]
    destination_node = route[-1]
    origin_destination_lats = (G.node[origin_node]['y'], G.node[destination_node]['y'])
    origin_destination_lons = (G.node[origin_node]['x'], G.node[destination_node]['x'])
    #ax.scatter(origin_destination_lons, origin_destination_lats, s=100,
     #          c='r', alpha=0.3, edgecolor='none', zorder=4)

    origin_lats = (G.node[origin_node]['y'])
    dest_lats = (G.node[destination_node]['y'])
    origin_lons = (G.node[origin_node]['x'])
    dest_lons = (G.node[destination_node]['x'])

    ax.scatter(origin_lons, origin_lats, s=100, c='r', alpha=0.5, edgecolor='none', zorder=4)
    ax.scatter(dest_lons, dest_lats, s=100, c='g', alpha=0.5, edgecolor='none', zorder=4)

    edge_nodes = list(zip(route[:-1], route[1:]))
    lines = []
    for u, v in edge_nodes:
        data = min([data for data in G.edge[u][v].values()], key=lambda x: x['length'])
        if 'geometry' in data:
            xs, ys = data['geometry'].xy
            lines.append(list(zip(xs, ys)))
        else:
            x1 = G.node[u]['x']
            y1 = G.node[u]['y']
            x2 = G.node[v]['x']
            y2 = G.node[v]['y']
            line = [(x1, y1), (x2, y2)]
            lines.append(line)
    lc = LineCollection(lines, colors='r', linewidths=4, alpha=0.3, zorder=3)
    ax.add_collection(lc)

