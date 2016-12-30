# Standard modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.optimize as opt
import scipy.special as spc
import utilities

verbose = True
plot_on = False
set_seed = True

def sample_polar_laplace(epsilon, size):
    # Sample from 2D Laplace distribution
    theta = np.random.uniform(0, 2 * np.pi, size=size)
    aux = np.random.uniform(0,1, size=size)

    # Compute -1 branch of Lambbert function
    W = spc.lambertw((aux-1)/np.e, -1)
    radius = np.real(-(1/epsilon) * (W +1))

    return radius, theta


def polar2euclid(radius, theta):
    aux = np.empty((radius.shape[0], 2))
    aux[:,0] = np.cos(theta)
    aux[:,1] = np.sin(theta)
    return aux * np.expand_dims(radius, axis=1)


# Test function

# Initialize
epsilon = 0.2
x0 = np.array([5,3], ndmin=2)
num_samples = 1000
x = np.empty((num_samples,2))

#for i in range(num_samples):
radius, theta = sample_polar_laplace(epsilon, num_samples)
x = x0 + polar2euclid(radius, theta)

fig = plt.figure(figsize=(4,4))
plt.scatter(x[:,0], x[:,1], alpha=0.4)
plt.axis('equal')
plt.show()


