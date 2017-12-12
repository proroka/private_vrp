
import numpy as np
import bitlib
import scipy.optimize
import time

BOUND_INF = 0
BOUND_HUNGARIAN = 1


def get_optimal_assignment(route_length_samples, use_initial_hungarian=True,
                           use_bound=True,
                           refined_bound=True,
                           bound_initialization=BOUND_HUNGARIAN):
  cached_results = {}

  num_vehicles, num_passengers, _ = route_length_samples.shape

  # Run hungarian to get a bound.
  preassigned = [None] * num_passengers
  available_vehicles_binary = 2 ** num_vehicles - 1
  bound = [float('inf'), 0, 0]
  c = np.mean(route_length_samples, axis=2)
  row_ind, col_ind = scipy.optimize.linear_sum_assignment(c)
  if bound_initialization == BOUND_HUNGARIAN:
    bound[0] = np.sum(c[row_ind, col_ind])
  if use_initial_hungarian:
    for v, p in zip(row_ind, col_ind):
      preassigned[p] = v
      available_vehicles_binary &= ~(1 << v)

  # Get a lower bound on the costs for each passenger.
  # Best fictious cost is to assign all vehicles to all passengers.
  if refined_bound:
    best_cost_per_passenger = np.mean(np.min(route_length_samples, axis=0), axis=1)
    remaining_best_cost = np.cumsum(best_cost_per_passenger[::-1])[::-1]
  else:
    remaining_best_cost = np.zeros(num_passengers)

  if isinstance(bound_initialization, float):
    bound[0] = bound_initialization

  def min_cost(passenger_index, available_vehicles_binary, cost_so_far):
    if (passenger_index, available_vehicles_binary) in cached_results:
      cost, solution = cached_results[(passenger_index, available_vehicles_binary)]
      bound[0] = min(cost + cost_so_far, bound[0])
      return cost, solution

    # Terminating conditions.
    # We've assigned all passengers. Remaining cost is 0.
    if passenger_index >= num_passengers:
      bound[0] = min(cost_so_far, bound[0])
      return 0., []

    # We haven't assigned all passengers, but have no more vehicles available.
    if available_vehicles_binary == 0:
      return float('inf'), None

    # Bound.
    bound[2] += 1
    if use_bound and cost_so_far + remaining_best_cost[passenger_index] > bound[0]:
      # It is important NOT to store this in the cache.
      bound[1] += 1
      return float('inf'), None

    num_available_vehicles = bitlib.count_ones(available_vehicles_binary)
    minimum_cost = float('inf')
    best_solution = None
    # -1 because we need to assign at least one vehicle.
    for passenger_assignment_binary in xrange(2 ** num_available_vehicles - 1):
      new_availability, assigned_vehicle_indices = bitlib.combine_bits(
          available_vehicles_binary, passenger_assignment_binary, preassigned[passenger_index])
      c = np.mean(np.min(route_length_samples[assigned_vehicle_indices, passenger_index, :], axis=0))
      next_cost, next_solution = min_cost(passenger_index + 1, new_availability, cost_so_far + c)
      total_cost = next_cost + c
      if minimum_cost > total_cost:
        best_solution = [assigned_vehicle_indices] + next_solution
        minimum_cost = total_cost
    cached_results[(passenger_index, available_vehicles_binary)] = (minimum_cost, best_solution)
    return minimum_cost, best_solution

  ret = min_cost(0, available_vehicles_binary, 0.)
  print 'Bound: ', bound
  return ret

num_vehicles = 12
num_passengers = 4
num_samples = 100

np.random.seed(0)
route_length_samples = np.random.rand(num_vehicles, num_passengers, num_samples)

s = time.time()
print get_optimal_assignment(route_length_samples, True, use_bound=True,
                             refined_bound=True, bound_initialization=BOUND_HUNGARIAN)
print 'Time for opt in TEST: ', time.time() - s