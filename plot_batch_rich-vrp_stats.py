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

runs = [18] # run without optimal, 10 to 100 robots, 500 iter, 16x16 grid
#runs = [16, 17] # runs with optimal, 4 to 16 robots, 250 iter each, 16x16 grid
#runs = [19] # run where optimal is without initial assignment; 500 iter, 4 to 12 robots; 16x16 grid

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


#col = get_cmap(len(epsilons)*6)
if 18 in runs:
    col = ['orange','green','g','m','b','c']
elif 16 in runs:
    col = ['r','g','g','m','b','c']
elif 19 in runs:        
    col = ['r','green','g','m','b','c']


#col = ['r','purple','g','m','b','cyan']

#-------------------------------------
# Performance


results = ['Mean Waiting Time', 'Sampled Costs']
res = 0
normalize = True

for epsilon in epsilons:
    print 'Epsilon: ', epsilon
    fig = plt.figure(figsize=(5, 6), frameon=False)
    ax = plt.gca()
    ind = 0
    offset = 0

    if res == 0: items = waiting_time
    elif res == 1: items = sampled_cost
    
    hung_baseline = np.zeros((len(max_assignable_vehicles_list), num_iter))
    # Compute Hungarian value as offset for bound
    for algo, w_dict in items.iteritems():
        if HUN in algo:
            if str(int(epsilon)) not in algo:
                continue
            
            hung_uniform = []
            for max_assignable_vehicles, w in w_dict.iteritems():
                index = np.where(np.array(max_assignable_vehicles_list)==max_assignable_vehicles)
                for i, v in enumerate(w):
                    if normalize:
                        hung_uniform.append(1.)
                    else:
                        hung_uniform.append(np.mean(w))
                    hung_baseline[index, i] = np.mean(v)

            hung_mean = np.mean(hung_uniform)*np.ones(len(max_assignable_vehicles_list))
            hung_lower, hung_upper = err(hung_uniform)
            hung_lower = hung_lower * np.ones(len(max_assignable_vehicles_list))
            hung_upper = hung_upper * np.ones(len(max_assignable_vehicles_list))

    for algo, w_dict in items.iteritems():
        if TRUE in algo:
            true_uniform = []
            for max_assignable_vehicles, w in w_dict.iteritems():
                index = np.where(np.array(max_assignable_vehicles_list)==max_assignable_vehicles)
                for i, v in enumerate(w):
                    if normalize:
                        true_uniform.append(np.mean(v) / hung_baseline[index, i])
                    else:
                        true_uniform.extend(np.mean(v))

            true_mean = np.mean(true_uniform)*np.ones(len(max_assignable_vehicles_list))
            true_lower, true_upper = err(true_uniform)
            print true_lower
            true_lower = true_lower * np.ones(len(max_assignable_vehicles_list))
            true_upper = true_upper * np.ones(len(max_assignable_vehicles_list))



    for algo, w_dict in items.iteritems():

        m_values = np.zeros((len(max_assignable_vehicles_list),))
        l_values = np.zeros((len(max_assignable_vehicles_list),))
        u_values = np.zeros((len(max_assignable_vehicles_list),))


        for max_assignable_vehicles, w in w_dict.iteritems():
            # means over passengers per iter
            index = np.where(np.array(max_assignable_vehicles_list)==max_assignable_vehicles)
            means = np.zeros(num_iter)
            for j in range(num_iter):
                means[j] = np.mean(w[j])
                if normalize:
                    means[j] /= hung_baseline[index, j]                
            m_values[index] = np.mean(means)
            l_values[index], u_values[index] = err(means)

        if HUN in algo:
            plt.plot(np.array(max_assignable_vehicles_list), hung_mean, color='black', ls='--', lw=2.0, label=algo)
            #plt.errorbar(np.array(max_assignable_vehicles_list) + ind*offset, hung_mean, hung_std, color=col(ind), fmt='o')
        elif TRUE in algo:
            plt.plot(np.array(max_assignable_vehicles_list), true_mean, color='black', lw=2.0, label=algo)
            ax.fill_between(np.array(max_assignable_vehicles_list), true_lower, true_upper, facecolor='black', alpha=0.5)
        elif SG in algo and 18 not in runs: # plot set greedy only for long run
            continue
        else:
            plt.plot(np.array(max_assignable_vehicles_list), m_values, color=col[ind], lw=2.0, label=algo)
            #plt.errorbar(np.array(max_assignable_vehicles_list) + ind*offset, m_values, s_values, color=col(ind), fmt='o')
            ax.fill_between(np.array(max_assignable_vehicles_list), l_values, u_values, facecolor=col[ind], alpha=0.5)

        if OPT in algo:
            K = float(max_assignable_vehicles - num_passengers)
            #fac = (1 - ((K-1)/K)**K)
            fac = 1 / math.e 
            bound =  m_values * (1-fac) + fac * hung_mean
            plt.plot(np.array(max_assignable_vehicles_list), bound, color='k', lw=3.0, ls=':' ,label='bound')

        ind += 1

    if 16 in runs:
        plt.ylim((0.6, 1.05))
    elif 18 in runs: 
        plt.ylim((0.2, 1.1))
    elif 19 in runs:
        plt.ylim((0.6, 1.1))

    plt. legend()
    ax.grid(True)
    fig_filename = fig_fn_base + '_' + str(int(epsilon)) + '_curve.eps'
    plt.show(block=False)

    plt.savefig(fig_filename, format='eps', transparent=True, frameon=False)



# Close figures

raw_input('Hit ENTER to close figure')
plt.close()