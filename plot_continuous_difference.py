import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, DateFormatter
import msgpack
import time


filenames = {
    'Non-private': 'data/simulation_optimal.dat',
    'Private non-redundant': 'data/simulation_epsilon_0.02.dat',
    # 'Private multi-allocation (1)': 'data/simulation_epsilon_0.02_variable.dat',
    # 'Private multi-allocation (0.5)': 'data/simulation_epsilon_0.02_variable_50.dat',
    'Private redundant': 'data/simulation_epsilon_0.02_variable_150.dat',
    # 'Private multi-allocation (2.0)': 'data/simulation_epsilon_0.02_variable_200.dat',
}

colors = {
    'Non-private': '#e98c50',
    'Private non-redundant': '#33a74d',
    # 'Private multi-allocation (1)': 'b',
    # 'Private multi-allocation (0.5)': 'c',
    'Private redundant': '#d03237',
    # 'Private multi-allocation (2.0)': 'k',
}

order = [
    'Non-private',
    'Private redundant',
    'Private non-redundant',
    # 'Private multi-allocation (1)',
    # 'Private multi-allocation (0.5)',
    # 'Private multi-allocation (2.0)',
]


min_timestamp = time.mktime(datetime.date(2016, 6, 1).timetuple())
max_timestamp = min_timestamp + 24 * 60 * 60
batching_duration = 20
smoothing_window_mins = 10
smoothing_window_stride = 10


TIME = 0
AVAILABLE_TAXIS = 1
TOTAL_TAXIS = 2
REQUESTS = 3
DROPPED_REQUESTS = 4
WAITING_TIME = 5
WAITING_TIME_FULL = 6


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
    return batch_times, batch_num_available_taxis, batch_total_taxis, batch_num_requests, batch_dropped_requests, mean_times, batch_waiting_times


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
    x, y, sy = smooth_plot(times, values, window=int(smoothing_window_mins * 60. / batching_duration), stride=int(60. * smoothing_window_stride / batching_duration))
    t = [datetime.datetime.fromtimestamp(t) for t in x]
    plt.plot(t, y, color, lw=2, label=label)
    plt.fill_between(t, y + sy, y - sy, facecolor=color, alpha=0.5)


def create_time_figure(data, what, label):
    fig, ax = plt.subplots()
    if what == DROPPED_REQUESTS:
        print 'Number of dropped requests:'
    for k in order:
        v = data[k]
        plot_smooth_data(v[TIME], v[what], colors[k], k)
        if what == DROPPED_REQUESTS:
            print '  %s: %d ~= %g%%' % (k, np.sum(v[what]), np.sum(v[what]) / (11e6 / 30.) * 100.)
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


def create_bar_figure(data, what, label):
    common_times = None
    for k, v in data.iteritems():
        if common_times is None:
            common_times = set(v[TIME])
        else:
            common_times &= set(v[TIME])
            bar_values = []
    baseline = None
    for k in order:
        all_values = []
        v = data[k]
        for i, t in enumerate(v[TIME]):
            if t in common_times:
                all_values.extend(v[what][i])
        mean_value = np.mean(all_values)
        if baseline is None:
            baseline = mean_value
        bar_values.append(mean_value)
    fig, ax = plt.subplots()
    x_values = np.arange(len(bar_values))
    width = 0.8
    plt.bar(x_values - width / 2., bar_values, width=width, color='lightskyblue')
    font = {
        'family': 'sans-serif',
        'color':  'white',
        'weight': 'bold',
        'size': 24,
        'horizontalalignment': 'center',
        'verticalalignment': 'top',
    }
    for x, y in zip(x_values, bar_values):
        plt.text(x, y - baseline * 0.1, '%d%%' % ((y / baseline - 1.) * 100.), fontdict=font)
    plt.xticks(x_values, order)
    ax.set_ylabel(label)
    ax.set_ylim(bottom=0)
    plt.tight_layout()


def create_violin_figure(data, what, label, percentile_cut=5):
    common_times = None
    for k, v in data.iteritems():
        if common_times is None:
            common_times = set(v[TIME])
        else:
            common_times &= set(v[TIME])
    ordered_values = []
    print 'Waiting time performance:'
    for k in order:
        all_values = []
        v = data[k]
        for i, t in enumerate(v[TIME]):
            if t in common_times:
                all_values.extend(v[what][i])
        all_values = np.array(all_values)
        print '  %s: %.3f +- %.3f [s]' % (k, np.mean(all_values), np.std(all_values))
        # p_low = np.percentile(all_values, percentile_cut)
        # p_high = np.percentile(all_values, 100 - percentile_cut)
        # all_values = all_values[np.logical_and(all_values >= p_low, all_values <= p_high)]
        ordered_values.append(all_values)
    fig, ax = plt.subplots(figsize=(8, 6 * .7))
    sns.violinplot(data=ordered_values, cut=0, gridsize=1000, palette=[colors[k] for k in order])
    plt.xticks(range(len(order)), order)
    ax.set_ylim(bottom=0, top=600)
    ax.grid(True)
    plt.tight_layout()

data = {}
for k, v in filenames.iteritems():
    data[k] = load_data(v)

create_time_figure(data, WAITING_TIME, 'Waiting time')
filename = 'figures/simulation_waiting_time.eps'
plt.savefig(filename, format='eps', transparent=True, frameon=False)

create_time_figure(data, AVAILABLE_TAXIS, 'Available taxis')
filename = 'figures/simulation_taxis.eps'
plt.savefig(filename, format='eps', transparent=True, frameon=False)

create_time_figure(data, REQUESTS, 'Requests to serve')
filename = 'figures/simulation_requests.eps'
plt.savefig(filename, format='eps', transparent=True, frameon=False)

create_time_figure(data, DROPPED_REQUESTS, 'Dropped requests')
filename = 'figures/simulation_dropped_requests.eps'
plt.savefig(filename, format='eps', transparent=True, frameon=False)

create_bar_figure(data, WAITING_TIME_FULL, 'Average waiting time')
filename = 'figures/simulation_mean_waiting_time.eps'
plt.savefig(filename, format='eps', transparent=True, frameon=False)

import seaborn as sns
sns.reset_orig()
create_violin_figure(data, WAITING_TIME_FULL, 'Waiting time')
filename = 'figures/simulation_violin_waiting_time.eps'
plt.savefig(filename, format='eps', transparent=True, frameon=False)

plt.show(block=False)

raw_input('Hit ENTER to close figure')
plt.close()
