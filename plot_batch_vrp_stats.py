# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import collections
import msgpack
import scipy.stats as stats

# My modules
import utilities.graph as util_graph
import utilities.noise as util_noise
import utilities.vrp as util_vrp
import utilities.plot as util_plot
import utilities.probabilistic as util_prob
import manhattan.data as manh_data


#-------------------------------------
# Load data
filename = 'data/vrp_batch_real_s2.dat'

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
ds = np.array([0.0001, 0.0002, 0.0004, 0.0008])
fig = plt.figure(figsize=(12,6), frameon=False)


for nv, num_vehicles in enumerate(num_vehicles_list):
    if num_vehicles == 100: 
        continue
    print 'Num vehicles:', num_vehicles

    means = np.zeros(len(epsilons))
    stds = np.zeros(len(epsilons))
    percentiles = np.zeros((len(epsilons),2))

    for e, epsilon in enumerate(epsilons):
        w = waiting_time[num_vehicles]['subopt_%g' % epsilon]
        means[e] = np.mean(w)
        stds[e] = np.std(w)
        #means[e], cl[e], cu[e] = mean_confidence_interval(data, confidence=0.95)
        ci = stats.t.interval(alpha = 0.95,              # Confidence level
                 df= 24,                    # Degrees of freedom
                 loc = means[e],         # Sample mean
                 scale = stds[e])             # Standard deviation estimate

        
        percentiles[e,:] = [np.percentile(w, 5), np.percentile(w, 95)]

    plt.subplot(121)
    #plt.plot(epsilons, means, color=colors[nv],label='Vehicles: %d' % num_vehicles)
    plt.errorbar(epsilons, means, yerr=[percentiles[:,0], percentiles[:,1]], color=colors[nv], linewidth=2, label='Vehicles: %d' % num_vehicles)
    w_opt = waiting_time[num_vehicles]['optimal']
    plt.plot([min(epsilons), max(epsilons)], [np.mean(w_opt)]*2, color=colors[nv])

    plt.subplot(122)
    #plt.plot(epsilons, means/np.mean(w_opt)-1, color=colors[nv],label='Vehicles: %d' % num_vehicles)
    plt.plot(epsilons, (means-np.mean(w_opt))/np.mean(w_opt), color=colors[nv], linewidth=2, label='Vehicles: %d' % num_vehicles)

plt.subplot(121)
ax = plt.gca()
ax.set_xscale('log')

plt.subplot(122)
ax = plt.gca()
ax.set_xscale('log')

filename = 'figures/results_batch_stats.eps'

plt.legend()
plt.show(block=False)
raw_input('Hit ENTER to close figure')

plt.close()







