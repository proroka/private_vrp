# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def plot_waiting_time_distr(waiting_time, percentile, bins, filename, save_fig=False, max_value=None):
    a = 0.5
    fig = plt.figure(figsize=(6,6), frameon=False)
    hdata, hbins = np.histogram(waiting_time, bins=bins)
    hdata_max = np.max(hdata)
    plt.hist(waiting_time, bins=bins, color='blue', alpha=a)
    for i in range(len(percentile)):
        plt.plot([percentile[i], percentile[i]],[0, hdata_max], 'k--')
    if not max_value:
        max_value = np.max(waiting_time)
    plt.xlim([-1, max_value+1])
    plt.ylim([0, hdata_max])
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency')
    if save_fig:
        plt.savefig(filename, format='png', transparent=True)

    plt.show()