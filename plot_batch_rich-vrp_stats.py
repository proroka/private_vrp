# Standard modules
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import msgpack
import scipy as sp
import scipy.stats
import math


#-------------------------------------
# Load data

run = 6

# Simulation data and figures
filename = 'data/rich-vrp_batch_s' + str(run) + '.dat'
fig_fn_base = 'figures/rich-vrp_batch_s' + str(run)


with open(filename, 'rb') as fp:
    print 'Loading waiting times...'
    data = msgpack.unpackb(fp.read())
    waiting_time = data['waiting_time']
    num_iter = data['num_iter']
    epsilons = data['epsilons']
    num_vehicles_list = data['num_vehicles_list']
    num_passengers = data['num_passengers']
    sampled_cost = data['sampled_cost']

#num_vehicles_list = [4, 6, 8, 10, 12, 14]

# VRP algorithm variants
TRUE = 'true'
OPT = 'optimal'
SG = 'set-greedy'
EG = 'element-greedy'
HUN = 'hungarian'
RAND = 'random'

# Get colors
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


col = get_cmap(len(epsilons)*5) #['b','r','g','m','c','y']


#-------------------------------------
# Performance


results = ['Mean Waiting Time', 'Sampled Costs']
res = 1

for epsilon in epsilons:
    print 'Epsilon: ', epsilon
    fig = plt.figure(figsize=(6, 6), frameon=False)
    ind = 0

    if res == 0: items = waiting_time
    elif res == 1: items = sampled_cost
    
    hung = np.zeros((len(num_vehicles_list),))  
    # Compute Hungarian value as offset for bound
    for algo, w_dict in items.iteritems():
        if HUN in algo:
            if str(int(epsilon)) not in algo:
                continue
             
            for num_vehicles, w in w_dict.iteritems():
                index = np.where(np.array(num_vehicles_list)==num_vehicles)
                hung[index] = np.mean(w)

    for algo, w_dict in items.iteritems():
        if algo == RAND:
            continue
        if str(int(epsilon)) not in algo:
            continue

        m_values = np.zeros((len(num_vehicles_list),))
        s_values = np.zeros((len(num_vehicles_list),))


        for num_vehicles, w in w_dict.iteritems():
            # means over passengers per iter
            means = np.zeros(num_iter)
            for j in range(num_iter):
                means[j] = np.mean(w[j])
                
            index = np.where(np.array(num_vehicles_list)==num_vehicles)
            m_values[index] = np.mean(w)
            s_values[index] = np.std(means)
            #print '%d : %f' % (num_vehicles, np.mean(w))

        
        print '%s \n%s' % (algo, m_values)
        print '%s' % (s_values)
        plt.plot(num_vehicles_list, m_values, color=col(ind), marker='o', label=algo)
        plt.errorbar(num_vehicles_list, m_values, s_values, color=col(ind))

        if OPT in algo:
            K = float(num_vehicles - num_passengers)
            #fac = (1 - ((K-1)/K)**K)
            fac = 1 / math.e 
            bound =  m_values * (1-fac) + fac * hung
            plt.plot(num_vehicles_list, bound, color='k', label='bound')

        plt.title(results[res])
        ind += 1

    plt. legend()
    fig_filename = fig_fn_base + str(int(epsilon)) + '_' + str(res) + '_curve.eps'
    plt.show(block=False)

    plt.savefig(fig_filename, format='eps', transparent=True, frameon=False)



# Close figures

raw_input('Hit ENTER to close figure')
plt.close()