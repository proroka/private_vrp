# Standard modules
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import msgpack
import scipy as sp
import scipy.stats


#-------------------------------------
# Load data
filename = 'data/vrp_batch_real_repeats_v3.dat'

with open(filename, 'rb') as fp:
    print 'Loading waiting times...'
    data = msgpack.unpackb(fp.read())
    waiting_time = data['waiting_time']
    epsilons = data['epsilons']
    num_vehicles_list = data['num_vehicles_list']
    repeats = [0] + data['repeats']

#-------------------------------------
# Helper function
def get_bounds(data, confidence=0.95):
    a = np.array(data)
    n = len(data)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    l = m - h
    u = m + h
    return m, l, u

def get_colors(N):
    if N == 1:
        return ['r']
    if N == 2:
        return ['r', 'g']
    if N == 3:
        return ['r', 'g', 'b']
    if N == 4:
        return ['r', 'g', 'b', 'k']
    color_norm = matplotlib.colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = matplotlib.cm.ScalarMappable(norm=color_norm, cmap='spectral')
    colors = []
    for i in range(N):
        colors.append(scalar_map.to_rgba(i))
    return colors

#-------------------------------------
# Plot
print 'Plotting...'

for num_vehicles in num_vehicles_list:
    # Create a new plot for different number of vehicles.
    fig = plt.figure(figsize=(12, 5.5), frameon=False)
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
            means[e], lower_errors[e], upper_errors[e] = get_bounds(waiting_time[num_vehicles][key])
        mean_w_opt = np.mean(waiting_time[num_vehicles]['optimal'])
        if not present:
            continue

        plt.subplot(121)
        plt.plot(epsilons, means, color=colors[i], lw=2)
        plt.fill_between(epsilons, lower_errors, upper_errors, color=colors[i], alpha=0.5)

        plt.subplot(122)
        plt.plot(epsilons, ((means - mean_w_opt) / mean_w_opt) * 100., color=colors[i], linewidth=2, label='D = %d' % (repeat + 1))

    plt.subplot(121)
    ax = plt.gca()
    ax.set_xscale('log')
    plt.plot([min(epsilons), max(epsilons)], [mean_w_opt] * 2, 'k--')
    plt.xlim([min(epsilons), max(epsilons)])
    plt.ylim(bottom=0)
    plt.grid('on')
    plt.xlabel('Epsilon')
    plt.ylabel('Waiting time [s]')
    plt.subplot(122)
    ax = plt.gca()
    ax.set_xscale('log')
    plt.xlim([min(epsilons), max(epsilons)])
    plt.ylim(bottom=0)
    ticks, _ = plt.yticks()
    plt.yticks(ticks, ['%g%%' % t for t in ticks])
    plt.xlabel('Epsilon')
    plt.ylabel('Waiting time increase')
    plt.grid('on')
    plt.legend()

    filename = 'figures/waiting_time_increase_%d.eps' % num_vehicles
    plt.savefig(filename, format='eps', transparent=True, frameon=False)

plt.show(block=False)
raw_input('Hit ENTER to close figure')
plt.close()







