import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, DateFormatter
import msgpack
import time
from scipy.stats import gaussian_kde
from scipy.stats import kde
import scipy.stats as st
import matplotlib.colors

import timezone
import manhattan.data as manh_data
import utilities.graph as util_graph
import tqdm


redundancy_main = 'Redundant'
start_offset = 3 * 60 * 60  # 3 am.
end_offset = 3 * 60 * 60  # 21 pm.
truncate_data = False  # truncates waiting times
tz = timezone.USTimeZone(-5, 'Eastern',  'EST', 'EDT')

filenames = {
    # 'True': 'data/simulation_optimal_10min_8000max.dat',
    'Non-redundant': 'data/simulation_normal_100_10min_8000max.dat',
    'Redundant': 'data/simulation_normal_100_variable_150_10min_8000max.dat',
}

colors = {
    # 'True': 'blue',
    'Non-redundant': 'red',
    'Redundant': 'green',
}

cmaps = {
    # 'True': 'Greens',
    'Non-redundant': 'Reds',
    'Redundant': 'Greens',
}

order = [
    # 'True',
    'Non-redundant',
    'Redundant',
]


min_timestamp = 1464753600.  # 1st of June 2016 midnight NYC.
max_timestamp = min_timestamp + 24 * 60 * 60
batching_duration = 20
smoothing_window_mins = 5  # 10
smoothing_window_stride = 5  # 10
ignore_ride_distance = 300.
ignore_ride_duration = 20


TIME = 0
AVAILABLE_TAXIS = 1
TOTAL_TAXIS = 2
REQUESTS = 3
DROPPED_REQUESTS = 4
WAITING_TIME = 5
WAITING_TIME_FULL = 6
REDUNDANCY = 7

conf_int = True
def err(a):
    if conf_int:
        m = np.mean(a)
        u, v = st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a))
        return u.item() - m, m - v.item()
    else:
        s = np.std(a)
        return s, s


def load_data(filename):
    with open(filename, 'rb') as fp:
        data = msgpack.unpackb(fp.read())
    batch_times = data['batch_times']
    batch_num_available_taxis = data['batch_num_available_taxis']
    batch_total_taxis = data['batch_total_taxis']
    batch_num_requests = data['batch_num_requests']
    batch_dropped_requests = data['batch_dropped_requests']
    batch_waiting_times = data['batch_waiting_times']
    if truncate_data:
        for i, w in enumerate(batch_waiting_times):
            batch_waiting_times[i] = list(np.minimum(batch_waiting_times[i], 600.))

    if 'number_allocated' in data:
      batch_redundancy = data['number_allocated']
    else:
      print 'There is no number_allocated data (i.e., D)'
      batch_redundancy = [0.] * len(batch_times)
    batch_times = np.array(batch_times).astype(np.float32)
    batch_num_available_taxis = np.array(batch_num_available_taxis).astype(np.float32)
    batch_total_taxis = np.array(batch_total_taxis).astype(np.float32)
    batch_num_requests = np.array(batch_num_requests).astype(np.float32)
    batch_dropped_requests = np.array(batch_dropped_requests).astype(np.float32)
    batch_redundancy = np.array(batch_redundancy).astype(np.float32)
    mean_times = []
    for w in batch_waiting_times:
        mean_times.append(np.mean(w) if w else np.nan)
    mean_times = np.array(mean_times)
    return (batch_times, batch_num_available_taxis, batch_total_taxis, batch_num_requests,
            batch_dropped_requests, mean_times, batch_waiting_times, batch_redundancy)

graph = manh_data.LoadMapData(use_small_graph=False)
nearest_neighbor_searcher = util_graph.NearestNeighborSearcher(graph)
taxi_data = manh_data.LoadTaxiData(graph, must_recompute=False, binary_data_filename='data/taxi_0c1ab5b97af4bcb760cd539196576898.pickle')
requests_time = []
num_simultaneuous_requests = []
request_pxy = []
request_dxy = []
print 'Analyzing taxi data...'
for i, (start_time, end_time, u, v) in enumerate(tqdm.tqdm(zip(
        taxi_data['pickup_time'], taxi_data['dropoff_time'],
        taxi_data['pickup_xy'], taxi_data['dropoff_xy']), total=len(taxi_data['pickup_time']))):
    if end_time < min_timestamp: continue
    if start_time >= max_timestamp: break
    if end_time - start_time < ignore_ride_duration: continue
    # Ignore crazy nodes (or rides outside manhattan).
    u_node, du = nearest_neighbor_searcher.Search(u)
    if du > ignore_ride_distance:
        continue
    v_node, dv = nearest_neighbor_searcher.Search(v)
    if dv > ignore_ride_distance:
        continue

    # Process simultaneuous requests.
    if not requests_time:
        requests_time.append(start_time)
        num_simultaneuous_requests.append(1)
        request_pxy.append([u])
        request_dxy.append([v])
    elif start_time < requests_time[-1] + batching_duration:
        num_simultaneuous_requests[-1] += 1
        request_pxy[-1].append(u)
        request_dxy[-1].append(v)
    else:
        while start_time >= requests_time[-1] + batching_duration:
            requests_time.append(requests_time[-1] + batching_duration)
            num_simultaneuous_requests.append(0)
            request_pxy.append([])
            request_dxy.append([])
        num_simultaneuous_requests[-1] += 1
        request_pxy[-1].append(u)
        request_dxy[-1].append(v)
requests_time = np.array(requests_time)
num_simultaneuous_requests = np.array(num_simultaneuous_requests)


def nyc_time(timestamp):
    return tz.fromutc(datetime.datetime.utcfromtimestamp(timestamp).replace(tzinfo=tz))

def smooth_plot(x, y, window=30, stride=1):
    def rolling_window(a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    r = rolling_window(y, window)
    only_nan = np.sum(np.isnan(r), axis=-1) == r.shape[-1]
    r[only_nan, 0] = 0.
    return np.mean(rolling_window(x, window), axis=-1)[::stride], np.nanmean(r, axis=-1)[::stride], np.nanstd(r, axis=-1)[::stride]


def plot_smooth_data(times, values, color, label, legend_label, ax=None,
                     start_offset=0, end_offset=0):
    x, y, sy = smooth_plot(times, values, window=int(smoothing_window_mins * 60. / batching_duration), stride=int(60. * smoothing_window_stride / batching_duration))
    t = [nyc_time(t) for t in x]
    if ax is None:
      ax = plt.gca()
    ax.plot(t, y, color, lw=2, label=legend_label)
    ax.fill_between(t, y + sy, y - sy, facecolor=color, alpha=0.5)
    ax.xaxis.set_major_locator(HourLocator(byhour=range(0, 24, 4), tz=tz))
    ax.xaxis.set_minor_locator(HourLocator(tz=tz))
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M', tz=tz))
    ax.fmt_xdata = DateFormatter('%H:%M', tz=tz)
    ax.grid(True)
    ax.set_xlabel('Time')
    ax.set_ylabel(label)
    ax.set_xlim(left=nyc_time(min_timestamp + start_offset), right=nyc_time(max_timestamp - end_offset))


data = {}
for k, v in filenames.iteritems():
    data[k] = load_data(v)

# Redundancy plot.
fig, ax = plt.subplots(figsize=(8,3.5))
plot_smooth_data(data[redundancy_main][TIME], data[redundancy_main][REDUNDANCY], colors[redundancy_main], label='Redundancy',
                 legend_label=redundancy_main, start_offset=start_offset, end_offset=end_offset)
plt.ylim([0., 4.])
plt.legend()
plt.tight_layout()
# filename = 'figures/final_redundancy.eps'
# plt.savefig(filename, format='eps', transparent=True, frameon=False)

# Average waiting time over time slices.
slices = 4  # hours
means = {}
stds = {}
for k in order:
  means[k] = []
  stds[k] = []
  for s in range(0, 24, slices):
    current_t = s * 60 * 60 + min_timestamp
    flatten_times = []
    for t, w in zip(data[k][TIME], data[k][WAITING_TIME_FULL]):
      if t >= current_t and t < current_t + 4 * 60 * 60:
        flatten_times.extend(w)
    means[k].append(np.mean(flatten_times))
    stds[k].append(err(flatten_times))

fig, ax1 = plt.subplots(figsize=(8, 3.5))
ax2 = ax1.twinx()
xticks = []
for s in range(0, 24, slices):
  xticks.append('%02d:00-%02d:00' % (s, s + slices))
n = len(xticks)
ind = np.arange(n)
width = (1. - 0.2) / len(order)
for i, k in enumerate(order):
  ax1.bar(ind + i * width, means[k], width, color=colors[k], bottom=0, yerr=np.array(stds[k]).T, ecolor='black', capsize=0, lw=2)
ax1.set_xticks(ind + (len(order) * width) / 2.)
ax1.set_xticklabels(xticks)
ax1.set_ylim(bottom=0)
ax1.set_xlabel('Time')
ax1.set_ylabel('Waiting time')
ax1.yaxis.grid()
ax1.set_xlim(-0.2, n - 1 + (len(order) * width) / 2. + 0.2)
x, y, sy = smooth_plot(requests_time, num_simultaneuous_requests, window=int(smoothing_window_mins * 60. / batching_duration), stride=int(60. * smoothing_window_stride / batching_duration))
xlim = ax1.get_xlim()
x = (x - min_timestamp) / (max_timestamp - min_timestamp) * (xlim[1] - xlim[0]) + xlim[0]
ax2.plot(x, y, '#777777', lw=2)
ax2.fill_between(x, y, np.zeros_like(y), facecolor='#777777', alpha=0.5)
ax2.set_ylabel('Number of requests per batch')
plt.tight_layout()
# filename = 'figures/final_waiting_times.eps'
# plt.savefig(filename, format='eps', transparent=True, frameon=False)

# Density plots.
import seaborn as sns
sns.reset_orig()
agg_over_hours = 1
f, axes = plt.subplots(2, n, sharex=True, sharey=True, figsize=(2 * n, 2 * 2))
for ax, s in zip(axes[0], range(0, 24, slices)):
  current_t = s * 60 * 60 + min_timestamp
  flatten_xy = []
  for t, xy in zip(requests_time, request_pxy):
    if t >= current_t and t < current_t + agg_over_hours * 60 * 60:
      flatten_xy.extend(xy)
  flatten_xy = np.array(flatten_xy)
  sns.kdeplot(flatten_xy[:,0], flatten_xy[:,1], cmap='Reds', shade=True, shade_lowest=False, ax=ax)
  ax.axis('equal')
  ax.set_xlim((581855, 590292))
  ax.set_ylim((4505860, 4518520))
  ax.axes.get_xaxis().set_ticks([])
  ax.axes.get_yaxis().set_ticks([])
for ax, s in zip(axes[1], range(0, 24, slices)):
  current_t = s * 60 * 60 + min_timestamp
  flatten_xy = []
  for t, xy in zip(requests_time, request_dxy):
    if t >= current_t and t < current_t + agg_over_hours * 60 * 60:
      flatten_xy.extend(xy)
  flatten_xy = np.array(flatten_xy)
  sns.kdeplot(flatten_xy[:,0], flatten_xy[:,1], cmap='Blues', shade=True, shade_lowest=False, ax=ax)
  ax.axis('equal')
  ax.set_xlim((581855, 590292))
  ax.set_ylim((4505860, 4518520))
  ax.axes.get_xaxis().set_ticks([])
  ax.axes.get_yaxis().set_ticks([])
# filename = 'figures/final_density.eps'
# plt.savefig(filename, format='eps', transparent=True, frameon=False)

# Occupied ratio.
flatui = ["#222222"]
black_cmap = matplotlib.colors.ListedColormap(sns.color_palette(flatui).as_hex())
fig, ax = plt.subplots(figsize=(8, 4))
leg = []
for k in order:
  y = data[k][WAITING_TIME]
  x = (data[k][TOTAL_TAXIS] - data[k][AVAILABLE_TAXIS]) / data[k][TOTAL_TAXIS]
  i = np.logical_not(np.isnan(y))
  y = y[i]
  x = x[i]
  sns.kdeplot(x, y, cmap=cmaps[k], shade=True, shade_lowest=False)
  sns.kdeplot(x, y, cmap=black_cmap, shade_lowest=False)
  leg.append(plt.Line2D((0,1),(0,0), color=colors[k]))
ax.grid(True)
plt.xlim(left=0.4, right=1.)
plt.ylim(bottom=0., top=600.)
plt.xlabel('Occupied ratio')
plt.ylabel('Waiting time')
ax.legend(leg, order)
plt.tight_layout()

# filename = 'figures/final_occupied.eps'
# plt.savefig(filename, format='eps', transparent=True, frameon=False)

plt.show(block=False)

raw_input('Hit ENTER to close figure')
plt.close()
