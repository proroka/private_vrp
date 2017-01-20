# Standard modules
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, DateFormatter
import collections
import heapq
import osmnx as ox
import tqdm
import sys

# My modules
import utilities.graph as util_graph
import utilities.noise as util_noise
import utilities.vrp as util_vrp
import utilities.plot as util_plot
import utilities.probabilistic as util_prob
import manhattan.data as manh_data

num_vehicles = 5000
drop_passengers_after = 1200.  # 20 minutes.
plot_animation = True

# graph = manh_data.LoadMapData(use_small_graph=False)
graph = util_graph.create_grid_map(grid_size=20, edge_length=100., default_speed=10.)
nearest_neighbor_searcher = util_graph.NearestNeighborSearcher(graph)
# taxi_data = manh_data.LoadTaxiData(graph, must_recompute=False)
taxi_data = manh_data.LoadTaxiData(graph, synthetic_rides=True, must_recompute=False,
                                   num_synthetic_rides=10000, synthetic_ride_speed=10.)
# manh_data.UpdateEdgeTime(graph, taxi_data, nearest_neighbor_searcher, must_recompute=False)
route_lengths = manh_data.LoadShortestPathData(graph, must_recompute=False)


class Taxi(object):
    def __init__(self, identifier, initial_node):
        self.identifier = identifier
        self.position = initial_node
        self.dropoff_time = None
    def pickup(self, request, current_time):
        route_time = route_lengths[self.position][request.pickup] + route_lengths[self.position][request.dropoff]
        self.dropoff_time = current_time + route_time
        self.position = request.dropoff
        return self.dropoff_time - request.time  # Total waiting time.
    def available(self):
        return self.dropoff_time is None
    def __hash__(self):
        return hash(self.identifier)
    def __eq__(x, y):
        return x.identifier == y.identifier


class Request(object):
    def __init__(self, identifier, time, pickup, dropoff):
        self.identifier = identifier
        self.time = time
        self.pickup = pickup
        self.dropoff = dropoff
    def __hash__(self):
        return hash(self.identifier)
    def __eq__(x, y):
        return x.identifier == y.identifier


# Priority queue holding each occupied taxi and when they become available.
class PriorityQueue(object):
    def __init__(self):
        self._queue = []
    def push(self, taxi):
        assert taxi.dropoff_time is not None
        heapq.heappush(self._queue, (taxi.dropoff_time, taxi))
    def pop(self):
        return heapq.heappop(self._queue)[-1]
    def peek(self):
        return self._queue[0][-1]
    def pop_available(self, current_time):
        taxis = set()
        while self._queue and self.peek().dropoff_time < current_time:
            taxi = self.pop()
            taxi.dropoff_time = None
            taxis.add(taxi)
        return taxis
    def __len__(self):
        return len(self._queue)

ignore_ride_distance = 300.
ignore_ride_duration = 20
batching_duration = 20  # Batch requests (passenger pickups).
max_batches = None

# Initialization.
nearest_dropoff_nodes, dist = nearest_neighbor_searcher.Search(taxi_data['dropoff_xy'])
nearest_dropoff_nodes = nearest_dropoff_nodes[dist < ignore_ride_distance]
nodes_and_counts = np.array(zip(*collections.Counter(nearest_dropoff_nodes).items()))
taxi_initial_nodes = np.random.choice(nodes_and_counts[0, :], size=num_vehicles, replace=True, p=nodes_and_counts[1, :].astype(np.float32) / float(np.sum(nodes_and_counts[1, :])))
available_taxis = set(Taxi(i, n) for i, n in enumerate(taxi_initial_nodes))
occupied_taxis = PriorityQueue()
current_batch_requests = set()

# Holds the end time of each batch.
batch_times = [float(taxi_data['pickup_time'][0])]
# Holds the waiting times for each passenger that requested a taxi during that batch.
batch_waiting_times = [[]]
# Holds the number of available taxis at the end of the batch (these taxis will be dispatched).
batch_num_available_taxis = [num_vehicles]
batch_num_requests = [0]

current_taxi_ride = 0
end_batch_times = np.arange(taxi_data['pickup_time'][0] + batching_duration, taxi_data['pickup_time'][-1], batching_duration)
for num_batches, end_batch_time in enumerate(tqdm.tqdm(end_batch_times, total=max_batches if max_batches else len(end_batch_times))):
    if max_batches and num_batches >= max_batches:
        break

    # Gather all new requests.
    while taxi_data['pickup_time'][current_taxi_ride] < end_batch_time:
        start_time, end_time, u, v = taxi_data['pickup_time'][current_taxi_ride], taxi_data['dropoff_time'][current_taxi_ride], taxi_data['pickup_xy'][current_taxi_ride], taxi_data['dropoff_xy'][current_taxi_ride]
        current_taxi_ride += 1
        if end_time - start_time < ignore_ride_duration:
            continue
        u_node, du = nearest_neighbor_searcher.Search(u)
        if du > ignore_ride_distance:
            continue
        v_node, dv = nearest_neighbor_searcher.Search(v)
        if dv > ignore_ride_distance:
            continue
        current_batch_requests.add(Request(current_taxi_ride, start_time, u_node, v_node))

    # Taxis that finished their ride should become available.
    available_taxis |= occupied_taxis.pop_available(end_batch_time)
    batch_num_available_taxis.append(len(available_taxis))
    batch_num_requests.append(len(current_batch_requests))

    # Dispatch.
    ordered_taxis = list(available_taxis)
    ordered_requests = list(current_batch_requests)
    allocation_cost = util_vrp.get_allocation_cost(
        route_lengths, [t.position for t in ordered_taxis],
        [r.pickup for r in ordered_requests])
    _, taxi_indices, request_indices = util_vrp.get_routing_assignment(allocation_cost)
    # Update lists.
    waiting_times = []
    for taxi_index, request_index in zip(taxi_indices, request_indices):
        taxi = ordered_taxis[taxi_index]
        request = ordered_requests[request_index]
        waiting_times.append(taxi.pickup(request, end_batch_time))
        occupied_taxis.push(taxi)
        available_taxis.remove(taxi)
        current_batch_requests.remove(request)
    # Garbage collect request that are abandoned.
    requests_to_remove = []
    for request in current_batch_requests:
        if end_batch_time - request.time > drop_passengers_after:
            requests_to_remove.append(request)
            waiting_times.append(drop_passengers_after)
    for request in requests_to_remove:
        current_batch_requests.remove(request)

    # Store information.
    batch_waiting_times.append(waiting_times)
    batch_times.append(end_batch_time)

batch_times = np.array(batch_times)
batch_num_available_taxis = np.array(batch_num_available_taxis)
batch_num_requests = np.array(batch_num_requests)


def smooth_plot(x, y, window=30, stride=1):
    def rolling_window(a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return np.mean(rolling_window(x, window), axis=-1)[::stride], np.nanmean(rolling_window(y, window), axis=-1)[::stride], np.nanstd(rolling_window(y, window), axis=-1)[::stride]

fig, ax = plt.subplots()
x, y, sy = smooth_plot(batch_times, batch_num_available_taxis, window=int(3600 / batching_duration), stride=int(600 / batching_duration))
t = [datetime.datetime.fromtimestamp(t) for t in x]
plt.plot(t, y, 'g', lw=2)
plt.fill_between(t, y + sy, y - sy, facecolor='g', alpha=0.5)
ax.xaxis.set_major_locator(DayLocator())
ax.xaxis.set_minor_locator(DayLocator())
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
ax.fmt_xdata = DateFormatter('%Y-%m-%d')
ax.grid(True)
ax.set_xlabel('Time')
ax.set_ylabel('Available taxis')

fig, ax = plt.subplots()
x, y, sy = smooth_plot(batch_times, batch_num_requests, window=int(3600 / batching_duration), stride=int(600 / batching_duration))
t = [datetime.datetime.fromtimestamp(t) for t in x]
plt.plot(t, y, 'r', lw=2)
plt.fill_between(t, y + sy, y - sy, facecolor='r', alpha=0.5)
ax.xaxis.set_major_locator(DayLocator())
ax.xaxis.set_minor_locator(DayLocator())
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
ax.fmt_xdata = DateFormatter('%Y-%m-%d')
ax.grid(True)
ax.set_xlabel('Time')
ax.set_ylabel('Number of requests')

fig, ax = plt.subplots()
mean_times = []
for t, w in zip(batch_times, batch_waiting_times):
    mean_times.append(np.mean(w) if w else np.nan)
mean_times = np.array(mean_times)
x, y, sy = smooth_plot(batch_times, mean_times, window=int(3600 / batching_duration), stride=int(600 / batching_duration))
t = [datetime.datetime.fromtimestamp(t) for t in x]
plt.plot(t, y, 'b', lw=2)
plt.fill_between(t, y + sy, y - sy, facecolor='b', alpha=0.5)
ax.xaxis.set_major_locator(DayLocator())
ax.xaxis.set_minor_locator(DayLocator())
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
ax.fmt_xdata = DateFormatter('%Y-%m-%d')
ax.grid(True)
ax.set_xlabel('Time')
ax.set_ylabel('Average waiting time')

plt.show(block=False)

raw_input('Hit ENTER to close figure')
plt.close()
