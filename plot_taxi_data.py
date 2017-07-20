import datetime
import heapq
import numpy as np
import matplotlib.pylab as plt
from matplotlib.dates import HourLocator, DateFormatter
import tqdm
import time
import msgpack

# My modules
import utilities.graph as util_graph
import manhattan.data as manh_data

use_small_graph = False
use_real_taxi_data = True
must_recompute = False
filename = 'data/taxi_fleet.dat'

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
min_timestamp = time.mktime(datetime.date(2016, 6, 1).timetuple())
max_timestamp = min_timestamp + 24 * 60 * 60

time = []
num_taxis = []
requests_time = []
num_simultaneuous_requests = []
print 'Analyzing taxi data...'
for i, (start_time, end_time, u, v) in enumerate(tqdm.tqdm(zip(
        taxi_data['pickup_time'], taxi_data['dropoff_time'],
        taxi_data['pickup_xy'], taxi_data['dropoff_xy']), total=min(max_rides, len(taxi_data['pickup_time'])) if max_rides else len(taxi_data['pickup_time']))):
    if end_time < min_timestamp - 30 * 60: continue
    if start_time >= max_timestamp + 30 * 60: break
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


def smooth_plot(x, y, window=30, stride=1):
    def rolling_window(a, window):
        a = np.array(a)
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return np.mean(rolling_window(x, window), axis=-1)[::stride], np.nanmean(rolling_window(y, window), axis=-1)[::stride], np.nanstd(rolling_window(y, window), axis=-1)[::stride]


#fig, ax1 = plt.subplots()
fig, ax1 = plt.subplots(figsize=(8, 6 * .7))
ax2 = ax1.twinx()
colors = ['#e98c50', '#33a74d']

num_taxis = np.interp(requests_time, time, num_taxis)
x, y, sy = smooth_plot(requests_time, num_taxis, window=int(30 * 60 / batching_duration), stride=int(10 * 60 / batching_duration))
t = [datetime.datetime.fromtimestamp(t) for t in x]
ax1.plot(t, y, color=colors[0], linestyle='solid')
ax1.fill_between(t, y + sy, y - sy, facecolor=colors[0], alpha=0.5)

# Save x and y.
with open(filename, 'wb') as fp:
    fp.write(msgpack.packb({'num_taxis': y.tolist(), 'time': x.tolist()}))

x, y, sy = smooth_plot(requests_time, num_simultaneuous_requests, window=int(30 * 60 / batching_duration), stride=int(10 * 60 / batching_duration))
t = [datetime.datetime.fromtimestamp(t) for t in x]
ax2.plot(t, y, color=colors[1], linestyle='solid')
ax2.fill_between(t, y + sy, y - sy, facecolor=colors[1], alpha=0.5)
ax1.xaxis.set_major_locator(HourLocator(interval=4))
ax1.xaxis.set_minor_locator(HourLocator())
ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
ax1.fmt_xdata = DateFormatter('%H:%M')
ax1.grid(True)
ax1.set_ylim(bottom=0, top=5500)
ax1.set_xlim(left=datetime.datetime.fromtimestamp(min_timestamp), right=datetime.datetime.fromtimestamp(max_timestamp))
ax1.set_xlabel('T') # Time
ax1.set_ylabel('O', color=colors[0]) # Occupied taxis
for tl in ax1.get_yticklabels():
    tl.set_color(colors[0])
ax2.set_ylabel('R', color=colors[1]) # Number of requests per batch
ax2.set_ylim(bottom=0, top=400)
for tl in ax2.get_yticklabels():
    tl.set_color(colors[1])

filename = 'figures/manhattan_taxi_analysis.eps'
plt.savefig(filename, format='eps', transparent=True, frameon=False)

plt.show(block=False)

raw_input('Hit ENTER to close figure')
plt.close()
