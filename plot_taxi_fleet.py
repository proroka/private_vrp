import datetime
import heapq
import numpy as np
import matplotlib.pylab as plt
from matplotlib.dates import HourLocator, DateFormatter
import tqdm
import time
import msgpack


with open('data/taxi_fleet.dat', 'rb') as fp:
    print 'Loading taxi fleet...'
    data = msgpack.unpackb(fp.read())
    taxi_fleet_timestamps = np.array(data['time'])
    taxi_fleet_size = np.array(data['num_taxis'])


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

batching_duration = 20
x, y, sy = smooth_plot(taxi_fleet_timestamps, taxi_fleet_size, window=2, stride=1)
t = [datetime.datetime.fromtimestamp(t) for t in x]
ax1.plot(t, y, color=colors[0], linestyle='solid')
ax1.fill_between(t, y + sy, y - sy, facecolor=colors[0], alpha=0.5)
plt.show()
