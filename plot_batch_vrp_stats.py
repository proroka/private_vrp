# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import collections
import msgpack

# My modules
import utilities.graph as util_graph
import utilities.noise as util_noise
import utilities.vrp as util_vrp
import utilities.plot as util_plot
import utilities.probabilistic as util_prob
import manhattan.data as manh_data

#-------------------------------------
# Load data
filename = 'data/vrp_batch_test.dat'

with open(filename, 'rb') as fp:
    print 'Loading waiting times...'
    data = msgpack.unpackb(fp.read())
    waiting_time = data['waiting_time']
    epsilons = data['epsilons']
    num_vehicles_list = data['num_vehicles_list']


#-------------------------------------
# Plot
print 'Plotting...'

colors = ['red', 'green', 'blue', 'cyan']
fig = plt.figure(figsize=(6,6), frameon=False)

for nv, num_vehicles in enumerate(num_vehicles_list):
    print 'Num vehicles:', num_vehicles
    means = np.zeros(len(epsilons))

    for e, epsilon in enumerate(epsilons):
        w = waiting_time[num_vehicles]['subopt_%g' % epsilon]
        means[e] = np.mean(w)

    plt.plot(epsilons, means, color=colors[nv])

ax = plt.gca()
ax.set_xscale('log')

filename = 'figures/results_batch_stats.eps'


plt.show(block=False)
raw_input('Hit ENTER to close figure')

plt.close()
