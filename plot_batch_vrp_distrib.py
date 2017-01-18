# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.optimize as opt
import scipy.special as spc
import pickle
import collections
from collections import Counter
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
filename = 'data/vrp_batch_real_distrib_s4.dat'


with open(filename, 'rb') as fp:
    print 'Loading waiting times...'
    data = msgpack.unpackb(fp.read())
    waiting_time = data['waiting_time']
    epsilons = data['epsilons']
    num_vehicles = data['num_vehicles']

# Indeces
OPT = 'optimal'
RAND = 'random'

#-------------------------------------

# Plot
print 'Plotting...'

set_x_lim = 500
set_y_lim = 0.25

max_value = max(np.max(w) for i, w in waiting_time.iteritems() if i != RAND)
num_bins = 80
for i, w in waiting_time.iteritems():
    print 'Mean, %s: %g' % (i, np.mean(w))
    if i == RAND:
            continue
    fig = plt.figure(figsize=(6,6), frameon=False)
    bins = np.linspace(-0.5, max_value+0.5, num_bins+1)
    stats = [np.mean(w)]
    fig_filename = 'figures/results_batch_real_distrib_%s.eps' % i
    util_plot.plot_waiting_time_distr(w, stats, bins, fig=fig, filename=fig_filename, max_value=max_value, set_x_max=set_x_lim, set_y_max=set_y_lim)


plt.show(block=False)
raw_input('Hit ENTER to close figure')

plt.close()

