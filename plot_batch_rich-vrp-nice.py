import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import msgpack
import seaborn as sns

runs = [11, 12]

run_index_offset = 0
dataframes = []
for r in runs:
    filename = 'data/rich-vrp_batch_s' + str(r) + '.dat'

    with open(filename, 'rb') as fp:
        print 'Loading waiting times...'
        data = msgpack.unpackb(fp.read())
        waiting_time = data['waiting_time']
        num_iter = data['num_iter']
        epsilons = data['epsilons']
        max_assignable_vehicles_list = data['max_assignable_vehicles_list']
        num_vehicles = data['num_vehicles']
        num_passengers = data['num_passengers']
        sampled_cost = data['sampled_cost']

    data = []
    column_names = ('algorithm', 'epsilon', 'max_assignable_vehicles', 'iteration', 'passenger', 'waiting_time')
    for algorithm_name, d in waiting_time.iteritems():
        tokens = algorithm_name.split('_', 3)
        if len(tokens) == 1:
            algorithm = tokens[0]
            epsilon = None
        else:
            print tokens
            algorithm, epsilon, _ = tokens
            epsilon = float(epsilon)
        for max_assignable_vehicles, run_values in d.iteritems():
            for run_index, waiting_times in enumerate(run_values):
                for p, waiting_time in enumerate(waiting_times):
                    data.append((algorithm, epsilon, max_assignable_vehicles, run_index + run_index_offset, p, waiting_time))
            run_index_offset += len(run_values)

    dataframes.append(pd.DataFrame(data, columns=column_names))

df = pd.concat(dataframes)

# sns.violinplot(data=df, x='algorithm', y='waiting_time')
#sns.tsplot(data=df, time='max_assignable_vehicles',
#           condition='algorithm', value='waiting_time')
plt.show()