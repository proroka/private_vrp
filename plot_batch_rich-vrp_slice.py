# Standard modules
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import msgpack
import scipy as sp
import scipy.stats as st
import math


#-------------------------------------
# Load data

runs = [30]

conf_int = True # return confidence interval or else std dev
normalize = False


# VRP algorithm variants
TRUE = 'true'
OPT = 'optimal'
SG = 'set-greedy'
EG = 'element-greedy'
HUN = 'hungarian'
RAND = 'random'

waiting_time = None
num_iter = 0
for r in range(0,len(runs)):
    run = runs[r]
    # Simulation data and figures
    filename = 'data/rich-vrp_batch_s' + str(run) + '.dat'
    fig_fn_base = 'figures/rich-vrp_batch_s' + str(run)

    with open(filename, 'rb') as fp:
        print 'Loading waiting times...'
        data = msgpack.unpackb(fp.read())
        epsilons = data['epsilons']
        max_assignable_vehicles_list = data['max_assignable_vehicles_list']
        num_vehicles = data['num_vehicles']
        num_passengers = data['num_passengers']
        sampled_cost = data['sampled_cost']

        num_iter += data['num_iter']
        if waiting_time is None:
            waiting_time = data['waiting_time']
        else:
            for algo, w_dict in data['waiting_time'].iteritems():
                for max_assignable_vehicles, w in w_dict.iteritems():
                    waiting_time[algo][max_assignable_vehicles].extend(w)
                    assert len(waiting_time[algo][max_assignable_vehicles]) == num_iter


# Get colors
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def err(a):
    if conf_int:
        u, v = st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a))
        return u.item(), v.item()
    else:
        m = np.mean(a)
        s = np.std(a)
        return m - s, m + s


col = {
  TRUE: 'black',
  OPT: 'red',
  SG: 'orange',
  EG: 'green',
  HUN: 'black',
  RAND: 'blue',
}


#-------------------------------------
# Performance

algorithms = sorted(list(set([a.rsplit('_', 1)[0] for a in waiting_time])))
epsilons = sorted(list(set([float(a.rsplit('_', 1)[1]) for a in waiting_time if not a.startswith(TRUE)])))
max_assignable_vehicles = waiting_time.values()[0].keys()[0]

fig = plt.figure(figsize=(10, 5), frameon=False)
ax = plt.gca()

mean_values = {}
for algo in algorithms:
  mean_values[algo] = []
  lower_values = []
  upper_values = []

  for epsilon in epsilons:
    reference = '%s_%g' % (HUN, epsilon)
    if algo == TRUE:
      current = algo
    else:
      current = '%s_%g' % (algo, epsilon)

    values = []
    for wb, w in zip(waiting_time[reference][max_assignable_vehicles], waiting_time[current][max_assignable_vehicles]):
      baseline = np.mean(wb)
      if normalize:
        values.append(np.mean(w) / baseline)
      else:
        values.append(np.mean(w))
    mean_values[algo].append(np.mean(values))
    l, u = err(values)
    lower_values.append(l)
    upper_values.append(u)

  print algo, col[algo]
  plt.plot(epsilons, mean_values[algo], '--' if algo == HUN else '-', color=col[algo], lw=2.0, label=algo, marker='o', ms=8.0)
  ax.fill_between(epsilons, lower_values, upper_values, facecolor=col[algo], alpha=0.5)

# Add bound if we can.
if OPT in mean_values and HUN in mean_values:
  fac = 1. / np.e
  values = (1. - fac) * mean_values[OPT] + (fac) * mean_values[HUN]
  plt.plot(epsilons, values, ':', color='black', lw=2.0, label='Bound', marker='o', ms=8.0)


box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.grid(True)

fig_filename = fig_fn_base + '_' + str(int(epsilon)) + '_curve.eps'
plt.show(block=False)

plt.savefig(fig_filename, format='eps', transparent=True, frameon=False)




# Close figures

raw_input('Hit ENTER to close figure')
plt.close()
