# Standard modules
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, DateFormatter
import collections
import heapq
import time
import tqdm
import msgpack

# My modules
import utilities.graph as util_graph
import utilities.noise as util_noise
import utilities.vrp as util_vrp
import manhattan.data as manh_data

num_vehicles = 8000  # Was 6000 originally.
drop_passengers_after = 600.  # 10 minutes, was 20 minutes originally.
min_vehicle_fleet = 200  # Minimum number of vehicle in the fleet (was 0 originally).
min_timestamp = 1464753600.  # 1st of June 2016 midnight NYC.
max_timestamp = min_timestamp + 24 * 60 * 60
version = 'normal'  # epsilon, normal, optimal.
version_info = '10min_8000max_200veh_nonoise'
epsilon = 0.02  # Only used when version is set to "epsilon".
sigma = 1.  # Only used when version is set to "normal".
algorithm = 'greedy'  # Only used when version is not "optimal".
allocate_extra_only_when = 1.5  # There must be 50% of vehicles left after assignment.
non_redundant = True  # Only used when version is set to "epsilon" or "normal".

if version == 'epsilon':
    version += '_%g' % epsilon
    if not non_redundant:
        version += '_variable_%d' % int(allocate_extra_only_when * 100)
if version == 'normal':
    version += '_%g' % sigma
    if not non_redundant:
        version += '_variable_%d' % int(allocate_extra_only_when * 100)

# If not None, the taxi fleet changes as a function of time (up to the specified number of vehicles: num_vehicles).
taxi_fleet_filename = 'data/taxi_fleet.dat'
# Adds extra taxis (by a given ratio, i.e. 1.1 -> 10% extra).
# According to http://www.nyc.gov/html/tlc/downloads/pdf/2014_taxicab_fact_book.pdf, the average ratio of occupied taxis is
# "On average, 64% of taxis are occupied during these hours (4PM - 6PM)." == 1.56x
extra_fleet = 1.56
fleet_window_in_secs = 60 * 10  # Fleet size is increased ahead of time and decrease after time (as a function of real taxi occupation)
dropoff_offset = 0  # Extra offset in time during which taxis are unavailable.

np.random.seed(1019)

graph = manh_data.LoadMapData(use_small_graph=False)
nearest_neighbor_searcher = util_graph.NearestNeighborSearcher(graph)
taxi_data = manh_data.LoadTaxiData(graph, must_recompute=False)
manh_data.UpdateEdgeTime(graph, taxi_data, nearest_neighbor_searcher, must_recompute=False)
route_lengths = manh_data.LoadShortestPathData(graph, must_recompute=False)
graph, route_lengths, nearest_neighbor_searcher = util_graph.normalize(graph, route_lengths)

if taxi_fleet_filename:
    with open(taxi_fleet_filename, 'rb') as fp:
        print 'Loading taxi fleet...'
        data = msgpack.unpackb(fp.read())
        taxi_fleet_timestamps = np.array(data['time'])
        taxi_fleet_size = np.array(data['num_taxis'])


class Taxi(object):
    def __init__(self, identifier, initial_node):
        self.identifier = identifier
        self.true_position = initial_node
        if version.startswith('epsilon'):
          radius, theta = util_noise.sample_polar_laplace(epsilon, 1)
          self.noise_offset = util_noise.polar2euclid(radius, theta)[0, :]
        elif version.startswith('normal'):
          self.noise_offset = np.random.normal(0., sigma, (2,))
        self.reported_dropoff_time = None
        self.dropoff_time = None
        self.is_immediately_available = True
        self.is_chosen = False
        self.update_reported_position()
    def pickup(self, request, current_time):
        # If the taxi is still dropping someone, adjust current_time to the future.
        # self.true_position holds the future dropoff location.
        if self.dropoff_time and self.dropoff_time > current_time:
            current_time = self.dropoff_time
            self.is_immediately_available = False
        else:
            self.is_immediately_available = True
        self.pickup_time = current_time + route_lengths[self.true_position][request.pickup]
        # Here, the taxi merely makes a promise to pickup the passenger at that time.
        request.add_taxi(self)
    def update_reported_position(self):
        if version.startswith('optimal'):
            self.reported_position = self.true_position
            return
        xy = util_graph.GetNodePosition(graph, self.true_position) + self.noise_offset
        nearest_node, _ = nearest_neighbor_searcher.Search(xy)
        self.reported_position = nearest_node
    def __hash__(self):
        return hash(self.identifier)
    def __eq__(x, y):
        if isinstance(x, int):
            return x == y.identifier
        if isinstance(y, int):
            return y == x.identifier
        return x.identifier == y.identifier


class Request(object):
    def __init__(self, identifier, time, pickup, dropoff):
        self.identifier = identifier
        self.time = time
        self.pickup = pickup
        self.dropoff = dropoff
        self.taxis = []
    def add_taxi(self, taxi):
        self.taxis.append(taxi)
    def arbitrate(self, current_time):
        taxi_ordered_by_pickup = sorted(self.taxis, key=lambda x: x.pickup_time)
        # The first taxi will pickup the passenger.
        first_taxi = taxi_ordered_by_pickup[0]
        first_taxi.dropoff_time = first_taxi.pickup_time + route_lengths[self.pickup][self.dropoff]
        first_taxi.true_position = self.dropoff
        old_reported_position = first_taxi.reported_position
        first_taxi.update_reported_position()
        first_taxi.reported_dropoff_time = (current_time + route_lengths[old_reported_position][self.pickup] +
                                            route_lengths[self.pickup][first_taxi.reported_position])
        first_taxi.is_chosen = True
        # Taxis that are not used go to the advertized position of the first taxi.
        first_taxi_xy = util_graph.GetNodePosition(graph, first_taxi.reported_position)
        for taxi in taxi_ordered_by_pickup[1:]:
            old_true_position = taxi.true_position
            if version.startswith('optimal'):
                taxi.true_position = first_taxi.reported_position
            else:
                taxi.true_position, _ = nearest_neighbor_searcher.Search(first_taxi_xy - taxi.noise_offset)
            if taxi.dropoff_time and taxi.dropoff_time > current_time:
                t = taxi.dropoff_time
            else:
                t = current_time
            taxi.dropoff_time = t + route_lengths[old_true_position][taxi.true_position]
            old_reported_position = taxi.reported_position
            taxi.reported_position = first_taxi.reported_position
            taxi.reported_dropoff_time = current_time + route_lengths[old_reported_position][self.pickup] + route_lengths[self.pickup][taxi.reported_position]
            taxi.is_chosen = False
        return first_taxi.pickup_time - self.time
    def __hash__(self):
        return hash(self.identifier)
    def __eq__(x, y):
        return x.identifier == y.identifier


# Priority queue holding each occupied taxi and when they become available.
class PriorityQueue(object):
    def __init__(self):
        self._queue = []
        self._set = set()
    def push(self, taxi):
        assert taxi.reported_dropoff_time is not None
        assert taxi not in self._set
        heapq.heappush(self._queue, (taxi.reported_dropoff_time + dropoff_offset, taxi))
        self._set.add(taxi)
    def pop(self):
        taxi = heapq.heappop(self._queue)[-1]
        self._set.remove(taxi)
        return taxi
    def peek_priority(self):
        return self._queue[0][0]
    def pop_available(self, current_time):
        taxis = set()
        while self._queue and self.peek_priority() < current_time:
            taxi = self.pop()
            taxis.add(taxi)
        return taxis
    def __len__(self):
        return len(self._queue)
    def __contains__(self, taxi):
        return taxi in self._set

def get_taxi_fleet_size(current_time):
    if taxi_fleet_filename:
        n = extra_fleet * np.interp(np.linspace(current_time - fleet_window_in_secs, current_time + fleet_window_in_secs, 10), taxi_fleet_timestamps, taxi_fleet_size)
        return int(max(min(num_vehicles, np.max(n)), min_vehicle_fleet))
    return num_vehicles

ignore_ride_distance = 300.
ignore_ride_duration = 20
batching_duration = 20  # Batch requests (passenger pickups).
max_batches = None

# Initialization.
nearest_dropoff_nodes, dist = nearest_neighbor_searcher.Search(taxi_data['dropoff_xy'])
nearest_dropoff_nodes = nearest_dropoff_nodes[dist < ignore_ride_distance]
nodes_and_counts = np.array(zip(*collections.Counter(nearest_dropoff_nodes).items()))
taxi_initial_nodes = np.random.choice(nodes_and_counts[0, :], size=num_vehicles, replace=True, p=nodes_and_counts[1, :].astype(np.float32) / float(np.sum(nodes_and_counts[1, :])))
taxi_count = get_taxi_fleet_size(min_timestamp)
max_taxi_identifier = taxi_count - 1
available_taxis = set(Taxi(i, n) for i, n in enumerate(taxi_initial_nodes[:taxi_count]))
occupied_taxis = PriorityQueue()
current_batch_requests = set()

current_taxi_ride = 0
num_batches = 0
end_batch_times = np.arange(taxi_data['pickup_time'][0] + batching_duration, taxi_data['pickup_time'][-1], batching_duration)
end_batch_times = end_batch_times[end_batch_times >= min_timestamp]
end_batch_times = end_batch_times[end_batch_times < max_timestamp]
end_batch_times = end_batch_times[:max_batches]

# Holds the end time of each batch.
batch_times = []
# Holds the waiting times for each passenger that requested a taxi during that batch.
batch_waiting_times = []
# Holds the number of available taxis at the end of the batch (these taxis will be dispatched).
batch_num_available_taxis = []
batch_total_taxis = []
batch_num_requests = []
batch_dropped_requests = []
availability_offsets = []
real_availability = []
number_allocated = []

for end_batch_time in tqdm.tqdm(end_batch_times):
    # Taxis that finished their ride should become available.
    newly_available_taxi = occupied_taxis.pop_available(end_batch_time)
    available_taxis |= newly_available_taxi
    for t in newly_available_taxi:
        max_taxi_identifier = max(t.identifier, max_taxi_identifier)

    # Increase fleet size if needed.
    new_taxi_count = get_taxi_fleet_size(end_batch_time)
    if new_taxi_count > taxi_count:
        for i in range(taxi_count, new_taxi_count):
            if i in occupied_taxis:
                continue
            max_taxi_identifier = max(max_taxi_identifier, i)
            if i in available_taxis:
                continue
            available_taxis.add(Taxi(i, taxi_initial_nodes[i]))
    taxi_count = new_taxi_count
    # Decrease fleet size if needed (taxis are only removed when available).
    if max_taxi_identifier >= taxi_count:
        for i in range(taxi_count, max_taxi_identifier + 1):
            available_taxis.discard(i)
        max_taxi_identifier = taxi_count - 1

    # Gather all new requests.
    while taxi_data['pickup_time'][current_taxi_ride] < end_batch_time:
        start_time, end_time, u, v = taxi_data['pickup_time'][current_taxi_ride], taxi_data['dropoff_time'][current_taxi_ride], taxi_data['pickup_xy'][current_taxi_ride], taxi_data['dropoff_xy'][current_taxi_ride]
        current_taxi_ride += 1
        if start_time < min_timestamp:
            continue
        if end_time - start_time < ignore_ride_duration:
            continue
        u_node, du = nearest_neighbor_searcher.Search(u)
        if du > ignore_ride_distance:
            continue
        v_node, dv = nearest_neighbor_searcher.Search(v)
        if dv > ignore_ride_distance:
            continue
        current_batch_requests.add(Request(current_taxi_ride, start_time, u_node, v_node))

    batch_num_available_taxis.append(len(available_taxis))
    batch_total_taxis.append(len(available_taxis) + len(occupied_taxis))
    batch_num_requests.append(len(current_batch_requests))

    # Dispatch.
    ordered_taxis = list(available_taxis)
    ordered_requests = list(current_batch_requests)
    if version.startswith('optimal'):
        # Taxis assigned per passenger.
        if current_batch_requests and available_taxis:
          number_allocated.append(float(min(len(available_taxis), len(current_batch_requests))) / float(len(current_batch_requests)))
          allocation_cost = util_vrp.get_allocation_cost(
              route_lengths, [t.reported_position for t in ordered_taxis],
              [r.pickup for r in ordered_requests])
          c, taxi_indices, request_indices = util_vrp.get_routing_assignment(allocation_cost)
        else:
          taxi_indices, request_indices = [], []
          if available_taxis and not current_batch_requests:
            number_allocated.append(float('nan'))
          elif not available_taxis and current_batch_requests:
            number_allocated.append(0.)
          else:
            number_allocated.append(float('nan'))
    else:
        if non_redundant:
            max_assignable_vehicles = min(len(available_taxis), len(current_batch_requests))
        else:
            max_assignable_vehicles = len(available_taxis) - int((allocate_extra_only_when - 1.) * len(current_batch_requests))
            if max_assignable_vehicles < len(current_batch_requests):
              max_assignable_vehicles = min(len(available_taxis), len(current_batch_requests))

        if ordered_requests and max_assignable_vehicles:
            vehicle_pos_noisy = util_graph.GetNodePositions(graph, [t.reported_position for t in ordered_taxis])
            passenger_node_ind = np.array([r.pickup for r in ordered_requests], dtype=np.uint32)
            route_length_samples = util_vrp.get_vehicle_sample_route_lengths(
                route_lengths, 100, vehicle_pos_noisy, passenger_node_ind, nearest_neighbor_searcher,
                epsilon if version.startswith('epsilon') else sigma,
                'laplace' if version.startswith('epsilon') else 'gauss')
            if algorithm == 'greedy':
                c, taxi_indices, request_indices = util_vrp.get_greedy_assignment(
                    route_length_samples, vehicle_pos_noisy, passenger_node_ind, max_assignable_vehicles,
                    epsilon if version.startswith('epsilon') else sigma,
                    'laplace' if version.startswith('epsilon') else 'gauss',
                    nearest_neighbor_searcher, graph)
                number_allocated.append(float(max_assignable_vehicles) / float(len(current_batch_requests)))
            else:
                repeat = max(0, int(float(max_assignable_vehicles) / float(len(current_batch_requests))) - 1)
                max_assignable_vehicles = repeat * len(current_batch_requests)
                number_allocated.append(float(repeat + 1))
                c, taxi_indices, request_indices, _ = util_vrp.get_repeated_routing_assignment(
                    route_length_samples, vehicle_pos_noisy, passenger_node_ind,
                    epsilon if version.startswith('epsilon') else sigma,
                    'laplace' if version.startswith('epsilon') else 'gauss',
                    nearest_neighbor_searcher, graph, repeat=repeat)
            assert max_assignable_vehicles == len(taxi_indices)
        else:
            taxi_indices, request_indices = [], []
            if max_assignable_vehicles and not ordered_requests:
              number_allocated.append(float('nan'))
            elif not max_assignable_vehicles and ordered_requests:
              number_allocated.append(0.)
            else:
              number_allocated.append(float('nan'))

    # Ask taxis to pickup.
    waiting_times = collections.defaultdict(lambda: np.inf)  # Waiting times for each request.
    for taxi_index, request_index in zip(taxi_indices, request_indices):
        taxi = ordered_taxis[taxi_index]
        request = ordered_requests[request_index]
        taxi.pickup(request, end_batch_time)
    # Arbitrate among taxis (when multiple allocations to the same request are made).
    for request_index in set(request_indices):
        request = ordered_requests[request_index]
        waiting_times[request] = request.arbitrate(end_batch_time)
        for taxi in request.taxis:
            if taxi.is_chosen:
                if taxi.is_immediately_available:
                    availability_offsets.append(taxi.reported_dropoff_time - taxi.dropoff_time)
                    real_availability.append(taxi.is_immediately_available)
            occupied_taxis.push(taxi)
            available_taxis.remove(taxi)
        current_batch_requests.remove(request)

    # Garbage collect request that are abandoned.
    requests_to_remove = []
    for request in current_batch_requests:
        if end_batch_time - request.time > drop_passengers_after:
            requests_to_remove.append(request)
            waiting_times[request] = drop_passengers_after
    batch_dropped_requests.append(len(requests_to_remove))
    for request in requests_to_remove:
        current_batch_requests.remove(request)

    # Store information.
    batch_waiting_times.append(waiting_times.values())
    batch_times.append(end_batch_time)

with open('data/simulation_%s_%s.dat' % (version, version_info), 'wb') as fp:
    fp.write(msgpack.packb({
        'batch_times': batch_times,
        'batch_num_available_taxis': batch_num_available_taxis,
        'batch_total_taxis': batch_total_taxis,
        'batch_num_requests': batch_num_requests,
        'batch_dropped_requests': batch_dropped_requests,
        'batch_waiting_times': batch_waiting_times,
        'availability_offsets': availability_offsets,
        'real_availability': real_availability,
        'number_allocated': number_allocated,
    }))


batch_times = np.array(batch_times)
batch_num_available_taxis = np.array(batch_num_available_taxis)
batch_total_taxis = np.array(batch_total_taxis)
batch_num_requests = np.array(batch_num_requests)
batch_dropped_requests = np.array(batch_dropped_requests)
number_allocated = np.array(number_allocated)


def smooth_plot(x, y, window=30, stride=1):
    def rolling_window(a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    r = rolling_window(y, window)
    only_nan = np.sum(np.isnan(r), axis=-1) == r.shape[-1]
    r[only_nan, 0] = 0.
    return np.mean(rolling_window(x, window), axis=-1)[::stride], np.nanmean(r, axis=-1)[::stride], np.nanstd(r, axis=-1)[::stride]


fig, ax = plt.subplots()
x, y, sy = smooth_plot(batch_times, batch_num_available_taxis, window=int(30 * 60 / batching_duration), stride=int(60 * 10 / batching_duration))
t = [datetime.datetime.fromtimestamp(t) for t in x]
plt.plot(t, y, 'g', lw=2, label='Available')
plt.fill_between(t, y + sy, y - sy, facecolor='g', alpha=0.5)
x, y, sy = smooth_plot(batch_times, batch_total_taxis, window=int(30 * 60 / batching_duration), stride=int(60 * 10 / batching_duration))
t = [datetime.datetime.fromtimestamp(t) for t in x]
plt.plot(t, y, 'b', lw=2, label='Total')
plt.fill_between(t, y + sy, y - sy, facecolor='b', alpha=0.5)
ax.xaxis.set_major_locator(HourLocator(interval=4))
ax.xaxis.set_minor_locator(HourLocator())
ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
ax.fmt_xdata = DateFormatter('%H:%M')
ax.grid(True)
ax.set_xlabel('Time')
ax.set_ylabel('Number of taxis')
ax.set_xlim(left=datetime.datetime.fromtimestamp(min_timestamp), right=datetime.datetime.fromtimestamp(max_timestamp))
ax.set_ylim(bottom=0)
plt.legend()
filename = 'figures/simulation_%s_taxis.eps' % version
plt.savefig(filename, format='eps', transparent=True, frameon=False)

fig, ax = plt.subplots()
x, y, sy = smooth_plot(batch_times, batch_num_requests, window=int(30 * 60 / batching_duration), stride=int(60 * 10 / batching_duration))
t = [datetime.datetime.fromtimestamp(t) for t in x]
plt.plot(t, y, 'b', lw=2, label='Total')
plt.fill_between(t, y + sy, y - sy, facecolor='b', alpha=0.5)
x, y, sy = smooth_plot(batch_times, batch_dropped_requests, window=int(30 * 60 / batching_duration), stride=int(60 * 10 / batching_duration))
t = [datetime.datetime.fromtimestamp(t) for t in x]
plt.plot(t, y, 'r', lw=2, label='Dropped')
plt.fill_between(t, y + sy, y - sy, facecolor='r', alpha=0.5)
ax.xaxis.set_major_locator(HourLocator(interval=4))
ax.xaxis.set_minor_locator(HourLocator())
ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
ax.fmt_xdata = DateFormatter('%H:%M')
ax.grid(True)
ax.set_xlabel('Time')
ax.set_ylabel('Number of requests per batch')
ax.set_xlim(left=datetime.datetime.fromtimestamp(min_timestamp), right=datetime.datetime.fromtimestamp(max_timestamp))
ax.set_ylim(bottom=0)
plt.legend()
filename = 'figures/simulation_%s_requests.eps' % version
plt.savefig(filename, format='eps', transparent=True, frameon=False)

fig, ax = plt.subplots()
mean_times = []
for w in batch_waiting_times:
    mean_times.append(np.mean(w) if w else np.nan)
mean_times = np.array(mean_times)
x, y, sy = smooth_plot(batch_times, mean_times, window=int(30 * 60 / batching_duration), stride=int(60 * 10 / batching_duration))
t = [datetime.datetime.fromtimestamp(t) for t in x]
plt.plot(t, y, 'b', lw=2)
plt.fill_between(t, y + sy, y - sy, facecolor='b', alpha=0.5)
ax.xaxis.set_major_locator(HourLocator(interval=4))
ax.xaxis.set_minor_locator(HourLocator())
ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
ax.fmt_xdata = DateFormatter('%H:%M')
ax.grid(True)
ax.set_xlabel('Time')
ax.set_ylabel('Average waiting time [s]')
ax.set_xlim(left=datetime.datetime.fromtimestamp(min_timestamp), right=datetime.datetime.fromtimestamp(max_timestamp))
ax.set_ylim(bottom=0)
filename = 'figures/simulation_%s_waiting_time.eps' % version
plt.savefig(filename, format='eps', transparent=True, frameon=False)

fig, ax = plt.subplots()
avg_offset = np.mean(availability_offsets)
p90 = np.percentile(availability_offsets, 90)
plt.hist(availability_offsets, bins=20)
l, t = plt.ylim()
plt.plot([avg_offset, avg_offset], [l, t], 'k--', lw=2)
plt.title('Average offset: %.3fs (90%% = %.3f) - Real availability: %d%%' % (avg_offset, p90, float(np.sum(real_availability)) / float(len(real_availability)) * 100.))
filename = 'figures/simulation_%s_availability.eps' % version
plt.savefig(filename, format='eps', transparent=True, frameon=False)

fig, ax = plt.subplots()
x, y, sy = smooth_plot(batch_times, number_allocated, window=int(30 * 60 / batching_duration), stride=int(60 * 10 / batching_duration))
t = [datetime.datetime.fromtimestamp(t) for t in x]
plt.plot(t, y, 'b', lw=2)
plt.fill_between(t, y + sy, y - sy, facecolor='b', alpha=0.5)
ax.xaxis.set_major_locator(HourLocator(interval=4))
ax.xaxis.set_minor_locator(HourLocator())
ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
ax.fmt_xdata = DateFormatter('%H:%M')
ax.grid(True)
ax.set_xlabel('Time')
ax.set_ylabel('Number of vehicles allocated per request')
ax.set_xlim(left=datetime.datetime.fromtimestamp(min_timestamp), right=datetime.datetime.fromtimestamp(max_timestamp))
ax.set_ylim(bottom=0)
filename = 'figures/simulation_%s_allocated.eps' % version
plt.savefig(filename, format='eps', transparent=True, frameon=False)

plt.show(block=False)

raw_input('Hit ENTER to close figure')
plt.close()
