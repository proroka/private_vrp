# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def plot_waiting_time_distr(waiting_time, percentile, bins, fig=None, filename=None, max_value=None, set_x_max=None, set_y_max=None):
    a = 0.5
    if not fig:
        fig = plt.figure(figsize=(6,6), frameon=False)
        
    hdata, hbins = np.histogram(waiting_time, bins=bins)
    # Normalize
    hdata = hdata / float(len(waiting_time))
    if set_y_max:
        hdata_max = set_y_max
    else:
        hdata_max = np.max(hdata)
    # Plot
    plt.bar(hbins[:-1], hdata, hbins[1]-hbins[0], bottom=None, hold=None, data=None)
    for i in range(len(percentile)):
        plt.plot([percentile[i], percentile[i]],[0, hdata_max], 'k--', linewidth=2.)
    if not max_value:
        max_value = np.max(waiting_time)
    plt.xlim([-1, max_value+1])
    if set_x_max:
        plt.xlim([-1, set_x_max])

    
    plt.ylim([0, hdata_max])
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency')
    plt.title(filename)
    if filename:
        plt.savefig(filename, format='eps', transparent=True, frameon=False)

    if not fig:
        plt.show()