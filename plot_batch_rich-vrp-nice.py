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
    num_iterations = None
    num_passengers = None
    column_names = ('algorithm', 'epsilon', 'max_assignable_vehicles', 'iteration', 'passenger', 'iteration_passenger', 'waiting_time')
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
            if num_iterations is not None:
                assert num_iterations == len(run_values)
            else:
                num_iterations = len(run_values)
            for run_index, waiting_times in enumerate(run_values):
                if num_passengers is not None:
                    assert num_passengers == len(waiting_times)
                else:
                    num_passengers = len(waiting_times)

                for p, waiting_time in enumerate(waiting_times):
                    iteration_index = run_index + run_index_offset
                    data.append((algorithm, epsilon, max_assignable_vehicles, iteration_index, p, p + iteration_index * num_passengers, waiting_time))
    
    run_index_offset += num_iterations
    dataframes.append(pd.DataFrame(data, columns=column_names))

df = pd.concat(dataframes, ignore_index=True)

# sns.violinplot(data=df, x='algorithm', y='waiting_time')
df = df[(df.epsilon == 100) | pd.isnull(df.epsilon)]
df = df.groupby('iteration').waiting_time.mean()

sns.tsplot(data=df, time='max_assignable_vehicles',
           value='waiting_time', condition='algorithm', unit='iteration')
plt.show()