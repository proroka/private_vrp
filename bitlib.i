%module bitlib

%{
#define SWIG_FILE_WITH_INIT

/* Put header files here or function declarations like below */
extern int count_ones(unsigned int v);
extern int msb(unsigned int x);
extern int combine_bits_wrapped(unsigned int mask, unsigned int v, int* vector, int n);
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (int* ARGOUT_ARRAY1, int DIM1) {(int* vector, int n)}
%apply (double* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(double* route_lengths, int num_vehicles, int num_passengers, int num_samples)}

%pythoncode %{
import numpy as np
def combine_bits(mask, v, add_extra_index=None):
  n = _bitlib.count_ones(mask)
  new_v, zeroed_positions = _bitlib.combine_bits_wrapped(mask, v, n)
  minimal_zeroed = zeroed_positions[zeroed_positions >= 0]
  if add_extra_index is not None:
    minimal_zeroed = np.append([add_extra_index], minimal_zeroed)
  return new_v, minimal_zeroed
%}

extern int count_ones(unsigned int v);
extern int msb(unsigned int x);
extern int combine_bits_wrapped(unsigned int mask, unsigned int v, int* vector, int n);
