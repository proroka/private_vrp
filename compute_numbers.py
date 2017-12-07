# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import collections
import time
import math
import itertools

numv = 9
nump = 3
red = numv/ nump

num_equal_partitions = math.factorial(numv) / (red)**nump

print 'Number of even partitions: ', num_equal_partitions

# All possible permutations of the set * all M-partitions for each permutation
# N! * [(N-1)!] / [M!(N-M)!]
num_total_partitions = math.factorial(numv) * math.factorial(numv-1) / (math.factorial(nump) * math.factorial(numv-nump))

print 'Total number of partitions: ', num_total_partitions

num_matchings = nump**numv

print 'Total number of matchings: ', num_matchings
