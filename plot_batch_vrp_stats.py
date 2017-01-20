# Standard modules
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import collections
import msgpack
import scipy as sp
import scipy.stats

# My modules
import utilities.graph as util_graph
import utilities.noise as util_noise
import utilities.vrp as util_vrp
import utilities.plot as util_plot
import utilities.probabilistic as util_prob
import manhattan.data as manh_data


#-------------------------------------
# Load data
filename = 'data/vrp_batch_real_repeats.dat'

with open(filename, 'rb') as fp:
    print 'Loading waiting times...'
    data = msgpack.unpackb(fp.read())
    waiting_time = data['waiting_time']
    epsilons = data['epsilons']
    num_vehicles_list = data['num_vehicles_list']
    repeats = [0] + data['repeats']


#-------------------------------------
# Helper function
def mean_confidence_interval(data, confidence=0.99):
    a = np.array(data)
    n = len(data)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m - h, m + h

def get_colors(N):
    color_norm = matplotlib.colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = matplotlib.cm.ScalarMappable(norm=color_norm, cmap='jet')
    colors = []
    for i in range(N):
        colors.append(scalar_map.to_rgba(i))
    return colors

#-------------------------------------
# Plot
print 'Plotting...'

for num_vehicles in num_vehicles_list:
    # Create a new plot for different number of vehicles.
    fig = plt.figure(figsize=(12, 6), frameon=False)
    colors = get_colors(len(repeats))
    print num_vehicles

    for i, repeat in enumerate(repeats):
        means = np.zeros(len(epsilons))
        upper_errors = np.zeros(len(epsilons))
        lower_errors = np.zeros(len(epsilons))
        present = True

        for e, epsilon in enumerate(epsilons):
            key = 'subopt_%g_%d' % (epsilon, repeat)
            if key not in waiting_time[num_vehicles]:
                present = False
                break
            means[e], lower_errors[e], upper_errors[e] = mean_confidence_interval(waiting_time[num_vehicles][key])
        mean_w_opt = np.mean(waiting_time[num_vehicles]['optimal'])
        if not present:
            continue

        plt.subplot(121)
        plt.plot(epsilons, means, color=colors[i], lw=2)
        plt.fill_between(epsilons, means, lower_errors, upper_errors, color=colors[i], alpha=0.5)
        plt.plot([min(epsilons), max(epsilons)], [mean_w_opt]*2, '--', color=colors[i])

        plt.subplot(122)
        plt.plot(epsilons, (means - mean_w_opt) / mean_w_opt, color=colors[i], linewidth=2, label='Repeat: %d' % repeat)

    plt.subplot(121)
    ax = plt.gca()
    ax.set_xscale('log')
    plt.subplot(122)
    ax = plt.gca()
    ax.set_xscale('log')
    plt.legend()
    plt.title('Number of vehicles: %d' % num_vehicles)

plt.show(block=False)
raw_input('Hit ENTER to close figure')
plt.close()







