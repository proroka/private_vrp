import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, DateFormatter
import msgpack
import time


filenames = {
    'Non-private': 'data/simulation_optimal.dat',
    'Private single allocation': 'data/simulation_epsilon_0.02.dat',
    'Private double allocation': 'data/simulation_epsilon_0.02_2.dat',
    'Private multi-allocation': 'data/simulation_epsilon_0.02_variable.dat',
}

colors = {
    'Non-private': 'r',
    'Private single allocation': 'g',
    'Private double allocation': 'b',
    'Private multi-allocation': 'k',
}

min_timestamp = time.mktime(datetime.date(2016, 6, 1).timetuple())
max_timestamp = min_timestamp + 24 * 60 * 60
batching_duration = 20


TIME = 0
AVAILABLE_TAXIS = 1
TOTAL_TAXIS = 2
REQUESTS = 3
WAITING_TIME = 5


def load_data(filename):
    with open(filename, 'rb') as fp:
        data = msgpack.unpackb(fp.read())
    batch_times = data['batch_times']
    batch_num_available_taxis = data['batch_num_available_taxis']
    batch_total_taxis = data['batch_total_taxis']
    batch_num_requests = data['batch_num_requests']
    batch_dropped_requests = data['batch_dropped_requests']
    batch_waiting_times = data['batch_waiting_times']
    batch_times = np.array(batch_times)
    batch_num_available_taxis = np.array(batch_num_available_taxis)
    batch_total_taxis = np.array(batch_total_taxis)
    batch_num_requests = np.array(batch_num_requests)
    batch_dropped_requests = np.array(batch_dropped_requests)
    mean_times = []
    for w in batch_waiting_times:
        mean_times.append(np.mean(w) if w else np.nan)
    mean_times = np.array(mean_times)
    return batch_times, batch_num_available_taxis, batch_total_taxis, batch_num_requests, batch_dropped_requests, mean_times


def smooth_plot(x, y, window=30, stride=1):
    def rolling_window(a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    r = rolling_window(y, window)
    only_nan = np.sum(np.isnan(r), axis=-1) == r.shape[-1]
    r[only_nan, 0] = 0.
    return np.mean(rolling_window(x, window), axis=-1)[::stride], np.nanmean(r, axis=-1)[::stride], np.nanstd(r, axis=-1)[::stride]


def plot_smooth_data(times, values, color, label):
    x, y, sy = smooth_plot(times, values, window=int(30 * 60 / batching_duration), stride=int(60 * 10 / batching_duration))
    t = [datetime.datetime.fromtimestamp(t) for t in x]
    plt.plot(t, y, color, lw=2, label=label)
    plt.fill_between(t, y + sy, y - sy, facecolor=color, alpha=0.5)


def create_time_figure(data, what, label):
    fig, ax = plt.subplots()
    for k, v in data.iteritems():
        print k
        plot_smooth_data(v[TIME], v[what], colors[k], k)
    ax.xaxis.set_major_locator(HourLocator(interval=4))
    ax.xaxis.set_minor_locator(HourLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.fmt_xdata = DateFormatter('%H:%M')
    ax.grid(True)
    ax.set_xlabel('Time')
    ax.set_ylabel(label)
    ax.set_xlim(left=datetime.datetime.fromtimestamp(min_timestamp), right=datetime.datetime.fromtimestamp(max_timestamp))
    ax.set_ylim(bottom=0)
    plt.legend()


data = {}
for k, v in filenames.iteritems():
    data[k] = load_data(v)

create_time_figure(data, WAITING_TIME, 'Waiting time')
create_time_figure(data, AVAILABLE_TAXIS, 'Available taxis')
create_time_figure(data, REQUESTS, 'Requests to serve')

plt.show(block=False)

raw_input('Hit ENTER to close figure')
plt.close()
