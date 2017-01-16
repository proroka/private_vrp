import datetime
import heapq
import matplotlib.pylab as plt
from matplotlib.dates import DayLocator, DateFormatter
import tqdm

# My modules
import utilities.graph as util_graph
import manhattan.data as manh_data

use_small_graph = False
use_real_taxi_data = True
must_recompute = False

graph = manh_data.LoadMapData(use_small_graph=use_small_graph)
nearest_neighbor_searcher = util_graph.NearestNeighborSearcher(graph)
taxi_data = manh_data.LoadTaxiData(graph, synthetic_rides=not use_real_taxi_data, must_recompute=must_recompute,
                                   num_synthetic_rides=1000, max_rides=1000000)


# Simple priority queue.
class PriorityQueue:
    def __init__(self):
        self._queue = []
    def push(self, item, priority=None):
        heapq.heappush(self._queue, (item if priority is None else priority, item))
    def pop(self):
        return heapq.heappop(self._queue)[-1]
    def peek(self):
        return self._queue[0][-1]
    def __len__(self):
        return len(self._queue)


# Keep ongoing taxi rides in priority queue.
queue = PriorityQueue()
ignore_ride_distance = 300.
ignore_ride_duration = 20
batching_duration = 20  # Batch requests (passenger pickups).
max_rides = None

time = []
num_taxis = []
requests_time = []
num_simultaneuous_requests = []
print 'Analyzing taxi data...'
for i, (start_time, end_time, u, v) in enumerate(tqdm.tqdm(zip(
        taxi_data['pickup_time'], taxi_data['dropoff_time'],
        taxi_data['pickup_xy'], taxi_data['dropoff_xy']), total=min(max_rides, len(taxi_data['pickup_time'])) if max_rides else len(taxi_data['pickup_time']))):
    if max_rides and i > max_rides: break
    if end_time - start_time < ignore_ride_duration: continue
    u_node, du = nearest_neighbor_searcher.Search(u)
    if du > ignore_ride_distance: continue
    v_node, dv = nearest_neighbor_searcher.Search(v)
    if dv > ignore_ride_distance: continue
    # Remove older cabs.
    while queue and queue.peek() <= start_time:
        t = queue.pop()
        time.append(t)
        num_taxis.append(len(queue))
    # Put new cab in.
    queue.push(end_time)
    time.append(start_time)
    num_taxis.append(len(queue))

    # Process simultaneuous requests.
    if not requests_time:
        requests_time.append(start_time)
        num_simultaneuous_requests.append(1)
    elif start_time < requests_time[-1] + batching_duration:
        num_simultaneuous_requests[-1] += 1
    else:
        while start_time >= requests_time[-1] + batching_duration:
            requests_time.append(requests_time[-1] + batching_duration)
            num_simultaneuous_requests.append(0)
        num_simultaneuous_requests[-1] += 1
# Drop remaining passengers.
while queue and queue.peek() <= start_time:
    t = queue.pop()
    time.append(t)
    num_taxis.append(len(queue))

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot_date([datetime.datetime.fromtimestamp(t) for t in time], num_taxis, 'b-')
ax2.plot_date([datetime.datetime.fromtimestamp(t) for t in requests_time], num_simultaneuous_requests, 'g-')
ax1.xaxis.set_major_locator(DayLocator())
ax1.xaxis.set_minor_locator(DayLocator())
ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
ax1.fmt_xdata = DateFormatter('%Y-%m-%d')
ax1.grid(True)
ax1.set_xlabel('Time')
ax1.set_ylabel('Occupied taxis', color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')
ax2.set_ylabel('Simultaneuous requests', color='g')
for tl in ax2.get_yticklabels():
    tl.set_color('g')

filename = 'figures/manhattan_taxi_analysis.eps'
plt.savefig(filename, format='eps', transparent=True, frameon=False)

plt.show(block=False)

raw_input('Hit ENTER to close figure')
plt.close()
