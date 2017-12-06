# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import collections
import time
import math

numv = 10
nump = 5
red = numv/ nump

num_equal_partitions = math.factorial(numv) / (red)**nump

print 'Number of even partitions: ', num_equal_partitions