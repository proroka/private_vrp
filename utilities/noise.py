import matplotlib.pylab as plt
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np
import os
import osmnx as ox
import pandas as pd
import pickle
import sys
import utm
from scipy.spatial import cKDTree
import scipy.special as spc


#-------------------------------------
# Functions

# Add noise to positions
def add_noise(point_locations, nearest_neighbor_searcher, epsilon, noise_model):
    
    # Compute noisy positions according to distribution
    if noise_model == 'gauss':
        noise_vector = np.random.normal(0.0, epsilon, (point_locations.shape[0],2))
        noisy_point_locations = point_locations + noise_vector

    elif noise_model == 'laplace': 
        radius, theta = sample_polar_laplace(epsilon, point_locations.shape[0])
        noise_vector = polar2euclid(radius, theta)
        noisy_point_locations = point_locations + noise_vector

    else: print 'Noise model not known.'

    # Find nearest node
    nearest_nodes, dist = nearest_neighbor_searcher.Search(noisy_point_locations)

    return nearest_nodes, noisy_point_locations

# Sample polar Laplacian noise
def sample_polar_laplace(epsilon, size):
    # Sample from 2D Laplace distribution
    theta = np.random.uniform(0, 2 * np.pi, size=size)
    aux = np.random.uniform(0,1, size=size)

    # Compute -1 branch of Lambbert function
    W = spc.lambertw((aux-1)/np.e, -1)
    radius = np.real(-(1/epsilon) * (W +1))

    return radius, theta

# Convert polar coordinates to Euclidean
def polar2euclid(radius, theta):
    aux = np.empty((radius.shape[0], 2))
    aux[:,0] = np.cos(theta)
    aux[:,1] = np.sin(theta)
    return aux * np.expand_dims(radius, axis=1)



