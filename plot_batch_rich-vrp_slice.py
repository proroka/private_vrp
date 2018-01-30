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



#-------------------------------------
# Performance

fig = plt.figure(figsize=(10, 5), frameon=False)
ax = plt.gca()

for epsilon in epsilons:
    print 'Epsilon: ', epsilon

    items = waiting_time

    for algo, w_dict in items.iteritems():
        if str(int(epsilon)) not in algo:
            continue

        m_values = np.zeros((len(max_assignable_vehicles_list),))
        l_values = np.zeros((len(max_assignable_vehicles_list),))
        u_values = np.zeros((len(max_assignable_vehicles_list),))


        for max_assignable_vehicles, w in w_dict.iteritems():
            # means over passengers per iter
            index = np.where(np.array(max_assignable_vehicles_list)==max_assignable_vehicles)
            means = np.zeros(num_iter)
            for j in range(num_iter):
                means[j] = np.mean(w[j])
              
            m_values[index] = np.mean(means)
            l_values[index], u_values[index] = err(means)


        #yerr = [m_values[0]-l_values[0], m_values[0]+u_values[0]]
        yerr = []
        yerr.append(m_values.flatten() - l_values.flatten())
        yerr.append(m_values.flatten() + u_values.flatten())

        if HUN in algo:
            ax.bar(epsilon-3, m_values,width=5.,color='b',align='center')
            #plt.errorbar(epsilon-3, m_values, yerr)
        
        elif EG in algo:
            ax.bar(epsilon+3, m_values,width=5.,color='g',align='center')
            #plt.errorbar(epsilon+3, m_values, yerr=yerr, 'k')


ax.bar(epsilon-3, 1, width=5.,color='b',align='center',label='Hungarian')
ax.bar(epsilon+3, 1 ,width=5.,color='g',align='center', label='Greedy')
plt.legend()

ax.grid(True)
plt.xticks(epsilons)
fig_filename = fig_fn_base + '_' + str(int(epsilon)) + '_curve.eps'
plt.show(block=False)

plt.savefig(fig_filename, format='eps', transparent=True, frameon=False)




# Close figures

raw_input('Hit ENTER to close figure')
plt.close()