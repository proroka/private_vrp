# Standard modules
import collections
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.optimize as opt
import time
import itertools

import noise
import probabilistic

BIG_NUMBER = 1e7

# Allocation cost given precomputed route lengths
def get_allocation_cost(route_lengths, vehicle_node_ind, passenger_node_ind):
    assert isinstance(route_lengths, np.ndarray) and len(route_lengths.shape) == 2, (
        'This function requires a contiguous route length matrix. Use the graph_util.normalize() function.')

    num_vehicles = len(vehicle_node_ind)
    num_passengers = len(passenger_node_ind)
    allocation_cost = np.zeros((num_vehicles, num_passengers))
    for i in range(num_vehicles):
        start = vehicle_node_ind[i]
        for j in range(num_passengers):
            end = passenger_node_ind[j]
            # Compute cost of shortest path for all possible allocations
            allocation_cost[i, j] = route_lengths[start, end]

    return allocation_cost

# Assign vehicles to passengers
def get_routing_assignment(allocation_cost):
    row_ind, col_ind = opt.linear_sum_assignment(allocation_cost)
    cost = allocation_cost[row_ind, col_ind]

    return cost, row_ind, col_ind

# Random assignment
def get_rand_routing_assignment(allocation_cost):
    num_vehicles = allocation_cost.shape[0]
    num_passengers = allocation_cost.shape[1]
    max_allocations = min(num_vehicles, num_passengers)
    row_ind = np.random.choice(np.arange(num_vehicles), size=max_allocations, replace=False)
    col_ind = np.random.choice(np.arange(num_passengers), size=max_allocations, replace=False)
    cost = allocation_cost[row_ind, col_ind]

    return cost, row_ind, col_ind

def get_updated_allocation_cost(vehicle_available, passenger_vehicles, vehicle_distances):
    P = np.array(passenger_vehicles, dtype=np.int32)
    V = np.expand_dims(np.array(vehicle_available, dtype=np.int32), axis=1)
    P = np.broadcast_arrays(np.ones((len(vehicle_available), 1, 1)), P)[1]
    V = np.expand_dims(np.broadcast_arrays(np.ones((1, len(passenger_vehicles))), V)[1], axis=-1)
    Wi = np.expand_dims(np.arange(len(passenger_vehicles)), axis=-1)
    W = np.broadcast_arrays(np.ones((len(vehicle_available), 1, 1)), Wi)[1]
    PV = np.concatenate([P, V], axis=-1)
    distances = vehicle_distances[PV, W]
    allocation_cost = np.mean(np.min(distances, axis=-2), axis=-1)
    return allocation_cost

# Precompute all vehicles' random positions and the distance from every sample to every passenger.
# Route lenghts: matrix of route lengths from nodes to nodes
# Returns a list of matrices
def get_vehicle_sample_route_lengths(route_lengths, vehicle_pos_noisy, passenger_node_ind, nearest_neighbor_searcher, epsilon, noise_model):
    num_samples = 100
    vehicle_sample_distances = []
    for p in vehicle_pos_noisy:
        point_locations = np.ones((num_samples, 2)) * p

        vehicle_node_ind_noisy, _ = noise.add_noise(point_locations, nearest_neighbor_searcher, epsilon, noise_model)
        # Second argument is row vector; V is matrix that repeats vehicle indeces to match num passengers
        V = np.broadcast_arrays(np.ones((passenger_node_ind.shape[0], 1)), vehicle_node_ind_noisy)[1]
        # P is matrix of passenger nodes; shape is transpose of V
        P = np.broadcast_arrays(np.ones((vehicle_node_ind_noisy.shape[0], 1)), passenger_node_ind)[1]
        # Route lengths is matrix; route lengths of every sample to every passenger
        vehicle_sample_distances.append(route_lengths[V, P.T]) # row index: passenger; col index: sample
    # Transform list of matrices to 3d array: vehicle index, passenger index, sample index
    vehicle_sample_distances = np.array(vehicle_sample_distances)

    #print 'Shape of vehicle_sample_distances: ' vehicle_sample_distances.shape()

    return vehicle_sample_distances

# Returns list of available vehicles indeces; 
# Returns list of lists; for each passegner, a list of assigned vehicle indeces
def get_assigned_vehicles(num_vehicles, num_passengers, row_ind, col_ind):
    assigned_vehicles = []
    for _ in range(num_passengers):
        assigned_vehicles.append([])
    # Add vehicle node index to list
    for v, p in zip(row_ind, col_ind):
        assigned_vehicles[p].append(v)
    available_vehicles = list(set(range(num_vehicles)) - set(row_ind))

    return available_vehicles, assigned_vehicles


# Get assignment of M vehicles to M passengers (redundant vehicles remain unused)
def get_Hungarian_assignment(route_lengths, vehicle_pos_noisy, passenger_node_ind, epsilon, noise_model, nearest_neighbor_searcher, graph):
     # Assign first round optimally, with Hungarian method
    allocation_cost = probabilistic.get_allocation_cost_noisy(route_lengths, vehicle_pos_noisy, passenger_node_ind, epsilon, noise_model, nearest_neighbor_searcher, graph)
    cost, row_ind, col_ind = get_routing_assignment(allocation_cost)

    return cost, row_ind, col_ind

# TODO: should return an allocation cost matrix instead of a sum of costs
def get_optimal_assignment(route_lengths, vehicle_pos_noisy, passenger_node_ind, epsilon, noise_model, nearest_neighbor_searcher, graph):

    num_passengers = len(passenger_node_ind)
    num_vehicles = len(vehicle_pos_noisy)

    # Precompute route lengths from all passengers to samples of vehicle positions
    vehicle_sample_route_lengths = get_vehicle_sample_route_lengths(route_lengths, vehicle_pos_noisy, passenger_node_ind, nearest_neighbor_searcher, epsilon, noise_model)
    num_samples = vehicle_sample_route_lengths.shape[2]
    
    # Compute first assignment (Hungarian): at least
    allocation_cost = np.mean(vehicle_sample_route_lengths, 2)
    cost, row_ind, col_ind = get_routing_assignment(allocation_cost)

    num_vehicles_left = num_vehicles-len(row_ind)
    available_vehicles = list(set(range(num_vehicles)) - set(row_ind))
    # print 'Available vehciles: ', available_vehicles
    #print num_vehicles_left

    matchings = list(itertools.product(range(num_passengers), repeat=num_vehicles_left))
    min_matching_allocation_cost = BIG_NUMBER
    
    for i in range(len(matchings)):
        if (i%5000==0): 
            print 'Matching %d out of %d\n' % (i, len(matchings))
        matching_allocation_cost = 0.
        matching = np.array(matchings[i])

        for p in range(num_passengers):
            v_assigned2_p = []
            assigned = (p == matching)
            print assigned
            #print 'Assigned: ', assigned
            for k in range(len(assigned)):
                #print 'Assigned[k]: ', assigned[k]
                if assigned[k]: v_assigned2_p.append(available_vehicles[k])

            # Also add vehicle from initial Hungarian assignment
            v_assigned2_p.append(row_ind[np.where(col_ind==p)][0])
            #print 'assigned vehicles: ', v_assigned2_p
            matching_allocation_cost += np.mean(np.amin(vehicle_sample_route_lengths[np.array(v_assigned2_p), p, :], 0))
        
        #print 'Matching: %s has cost %f' %(matching, matching_allocation_cost)
        if matching_allocation_cost < min_matching_allocation_cost:
            #print 'New minimum found.'
            optimal_matching = matching
            min_matching_allocation_cost = matching_allocation_cost

    #print 'Cost: ', min_matching_allocation_cost
    #print 'Optimal matching: ', optimal_matching

    #print row_ind
    #print col_ind

    # Add to indeces
    available_vehicles = list(set(range(num_vehicles)) - set(row_ind))
    for l in range(len(optimal_matching)):
        row_ind = np.append(row_ind, available_vehicles[l])
        col_ind = np.append(col_ind, optimal_matching[l])

    #print row_ind
    #print col_ind

    return min_matching_allocation_cost, row_ind, col_ind






# Assign a redundant number of vehicles to each passenger; after first round is allocated, assign vehicles provide largest gain
# Gain: as measured by largest decrease in cost (history-dependent objective)
def get_greedy_assignment(route_lengths, vehicle_pos_noisy, passenger_node_ind, epsilon, noise_model, nearest_neighbor_searcher, graph):
    verbose = False


    # Assign first round optimally, with Hungarian method
    #allocation_cost = probabilistic.get_allocation_cost_noisy(route_lengths, vehicle_pos_noisy, passenger_node_ind, epsilon, noise_model, nearest_neighbor_searcher, graph)
    #cost, row_ind, col_ind = get_routing_assignment(allocation_cost)
    #print 'Number of assigned vehicles at start: ', len(row_ind)

    # Precompute route lengths from all passengers to samples of vehicle positions
    vehicle_sample_route_lengths = get_vehicle_sample_route_lengths(route_lengths, vehicle_pos_noisy, passenger_node_ind, nearest_neighbor_searcher, epsilon, noise_model)
    num_samples = vehicle_sample_route_lengths.shape[2]

    # Compute first assignment (Hungarian)
    allocation_cost = np.mean(vehicle_sample_route_lengths, 2)
    cost, row_ind, col_ind = get_routing_assignment(allocation_cost)
    if verbose: print 'Allocation cost: \n', allocation_cost

    redundant_vehicles = len(vehicle_pos_noisy) - len(passenger_node_ind)
    if verbose: print 'Redundant vehicles: ', redundant_vehicles
    #assert redundant_vehicles >= 0, ('No redundant vehicles: the number of vehicles must be larger than number of passengers.')

    # Assign remaining vehicles greedily; up to max number
    available_vehicles, assigned_vehicles = get_assigned_vehicles(len(vehicle_pos_noisy), len(passenger_node_ind), row_ind, col_ind)

    if verbose: print 'Currently assigned vehicles: ', assigned_vehicles

    # Compute updated allocation cost (as a function fo already assigned vechiles)
    updated_allocation_cost =  allocation_cost.copy() 
    for _ in range(redundant_vehicles):
        
        updated_allocation_cost = np.ones((len(vehicle_pos_noisy),len(passenger_node_ind))) * BIG_NUMBER
        if verbose: print 'updated cost should be BIG: \n', updated_allocation_cost

        for p in range(len(passenger_node_ind)):
            # Get indeces of currently assigned vehicles
            v_assigned2_p = assigned_vehicles[p]
            prev_cost_p = np.mean(np.amin(vehicle_sample_route_lengths[np.array(v_assigned2_p), p, :], 0))

            # Compute change when additing redundanct vehicle v
            for v in available_vehicles:
                v_indeces = v_assigned2_p[:]
                v_indeces.append(v)
                v_indeces = np.array(v_indeces)
                # Mean of minimum over all assigned vehicles 
                updated_allocation_cost[v, p] = np.mean(np.amin(vehicle_sample_route_lengths[v_indeces, p, :], 0)) - prev_cost_p

        if verbose: print 'Only for available vehicles ', available_vehicles
        if verbose: print 'Updated cost should be BIG except from available veh.: \n', updated_allocation_cost
        # Compute gain (previous minus current): reduction in waiting time when assigning additional v to p
        
        # Find indeces of minimum value
        #v_min_ind, p_min_ind = np.unravel_index(np.argmin(updated_allocation_cost), updated_allocation_cost.shape)
        v_min_ind, p_min_ind = np.unravel_index(np.argmin(updated_allocation_cost), updated_allocation_cost.shape)

        if verbose: print 'Will assign vehicle: %d  to passenger: %d' % (v_min_ind, p_min_ind)
        if verbose: print '... of available vehicles: ', available_vehicles

        # Update remaining vehicles
        row_ind = np.append(row_ind, v_min_ind)
        col_ind = np.append(col_ind, p_min_ind)

        if verbose: print 'Trying to remove %d from %s' %(v_min_ind, available_vehicles)
        available_vehicles.remove(v_min_ind)
        assigned_vehicles[p_min_ind].append(v_min_ind)

        #print 'Number of assigned vehicles at end of greedy: ', len(row_ind)
        #print len(col_ind)

    return updated_allocation_cost, row_ind, col_ind


def get_set_greedy_assignment(route_lengths, vehicle_pos_noisy, passenger_node_ind, epsilon, noise_model, nearest_neighbor_searcher, graph, repeats):

    cost, row_ind, col_ind, vd = get_repeated_routing_assignment(route_lengths, vehicle_pos_noisy, passenger_node_ind, epsilon, noise_model, nearest_neighbor_searcher, graph, repeat=0)
    
    previous_repeat = 0
    for repeat in repeats:
        cost, row_ind, col_ind, _ = get_repeated_routing_assignment(route_lengths, vehicle_pos_noisy, passenger_node_ind, epsilon, noise_model, nearest_neighbor_searcher, graph, repeat=repeat,
                                                                             previous_row_ind=row_ind, previous_col_ind=col_ind, previous_repeat=previous_repeat, previous_vehicle_distances=vd)
        previous_repeat = repeat


    return cost, row_ind, col_ind
    

def get_repeated_routing_assignment(route_lengths, vehicle_pos_noisy, passenger_node_ind, epsilon, noise_model, nearest_neighbor_searcher, graph, repeat=1,
                                    previous_row_ind=None, previous_col_ind=None, previous_repeat=None, previous_vehicle_distances=None):
    assert isinstance(route_lengths, np.ndarray) and len(route_lengths.shape) == 2, (
        'This function requires a contiguous route length matrix. Use the graph_util.normalize() function.')

    if len(vehicle_pos_noisy) == 0:
        return 0., [], [], None

    # Precompute all vehicles' random positions and the distance from every sample to every passenger.
    if previous_vehicle_distances is None:
        num_samples = 100
        vehicle_sample_distances = []
        for p in vehicle_pos_noisy:
            point_locations = np.ones((num_samples, 2)) * p
            vehicle_node_ind_noisy, _ = noise.add_noise(point_locations, nearest_neighbor_searcher, epsilon, noise_model)
            V = np.broadcast_arrays(np.ones((passenger_node_ind.shape[0], 1)), vehicle_node_ind_noisy)[1]
            P = np.broadcast_arrays(np.ones((vehicle_node_ind_noisy.shape[0], 1)), passenger_node_ind)[1]
            vehicle_sample_distances.append(route_lengths[V, P.T])
        vehicle_sample_distances = np.array(vehicle_sample_distances)
    else:
        vehicle_sample_distances = previous_vehicle_distances

    if len(passenger_node_ind) == 0:
        return 0., [], [], vehicle_sample_distances

    # Holds the mapping from passenger to allocated vehicles.
    already_allocated_vehicles = []
    for _ in passenger_node_ind:
        already_allocated_vehicles.append([])
    if previous_row_ind and previous_col_ind and previous_repeat is not None:
        for v, p in zip(previous_row_ind, previous_col_ind):
            already_allocated_vehicles[p].append(v)
        vehicle_indices = list(set(range(len(vehicle_pos_noisy))) - set(previous_row_ind))
        assert previous_repeat < repeat
    else:
        # Holds the indices of vehicles that are still available.
        vehicle_indices = range(len(vehicle_pos_noisy))
        previous_repeat = -1

    for _ in range(repeat - previous_repeat):
        # Compute cost matrix.
        allocation_cost = get_updated_allocation_cost(vehicle_indices, already_allocated_vehicles, vehicle_sample_distances)
        cost, row_ind, col_ind = get_routing_assignment(allocation_cost)
        # Update lists.
        for v, p in zip(row_ind, col_ind):
            already_allocated_vehicles[p].append(vehicle_indices[v])
        newly_allocated_vehicles = set(vehicle_indices[i] for i in row_ind)
        vehicle_indices = list(set(vehicle_indices) - newly_allocated_vehicles)
        if not vehicle_indices:
            break

    # Report final allocation.
    row_ind = []
    col_ind = []
    for p, vehicles in enumerate(already_allocated_vehicles):
        for v in vehicles:
            row_ind.append(v)
            col_ind.append(p)
    return cost, row_ind, col_ind, vehicle_sample_distances





# In: route_lengths: ndarray
def compute_waiting_times(route_lengths, vehicle_node_ind, passenger_node_ind, row_ind, col_ind):
    waiting_times = collections.defaultdict(lambda: np.inf)

    for v, p in zip(row_ind, col_ind):
        waiting_times[p] = min(waiting_times[p], route_lengths[vehicle_node_ind[v]][passenger_node_ind[p]])
        #waiting_times[p] = min(waiting_times[p], route_lengths[vehicle_node_ind[v], passenger_node_ind[p]])

    w = []
    for p in range(len(passenger_node_ind)):
        w.append(float(waiting_times[p]))
    return w


#####

def old_get_optimal_assignment(route_lengths, vehicle_pos_noisy, passenger_node_ind, epsilon, noise_model, nearest_neighbor_searcher, graph):

    cost = 0
    col_ind = []
    row_ind = []
    num_passengers = len(passenger_node_ind)
    num_vehicles = len(vehicle_pos_noisy)

    # Precompute route lengths from all passengers to samples of vehicle positions
    vehicle_sample_route_lengths = get_vehicle_sample_route_lengths(route_lengths, vehicle_pos_noisy, passenger_node_ind, nearest_neighbor_searcher, epsilon, noise_model)
    num_samples = vehicle_sample_route_lengths.shape[2]
    
    # Compute all possible permutations of vehicle IDs
    vehicle_permutations = list(itertools.permutations(range(len(vehicle_pos_noisy))))
    
    # Precompute a list of tuples with indeces specifying the separation of vehicles into M sets
    # The combinations are sorted in ascending order
    separation_indeces_list = list(itertools.combinations(range(num_vehicles-1), num_passengers-1))
    print 'Separations: ', separation_indeces_list

    partitions_list = []

    # For each permutation, compute all partitions
    for perm_ind in range(len(vehicle_permutations)):
        print 'Current vehicle permutation: ', vehicle_permutations[perm_ind]
        perm = vehicle_permutations[perm_ind]
        for s in range(len(separation_indeces_list)):
            separation_indeces = separation_indeces_list[s] # tuple of separation indeces

            # Generate partition of M sets of vehicles
            partition = []
            ind = 0
            for j in range(num_passengers-1):
                partition.append(perm[ind:separation_indeces[j]])
                ind = separation_indeces[j]
            partition.append(perm[ind:])
            print 'Partition: ', partition

            partitions_list.append(partition)
            # For each partition, compute allocation_cost matrix
            # For each set of vehicles (of partition), compute expected min. waiting time for each passenger
            # Compute optimal assignment
            # Store allocation cost, update min

    print 'Partitions list: ', partitions_list

    # Return partition and assignment indeces of assignment that resulted in minimum cost

    return cost, row_ind, col_ind