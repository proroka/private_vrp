/*
swig -python bitlib.i && \
gcc -c bitlib.c bitlib_wrap.c \
  -I/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/include/python2.7 \
  -I/usr/local/lib/python2.7/site-packages/numpy/core/include && \
ld -bundle -flat_namespace -undefined suppress -o _bitlib.so bitlib.o bitlib_wrap.o

On linux:
swig -python bitlib.i &&
gcc -O3 -fPIC -c bitlib.c bitlib_wrap.c -I /usr/include/python2.7
gcc -shared bitlib.o bitlib_wrap.o -o _bitlib.so
*/

static const unsigned char BitsSetTable256[256] =
{
#   define B2(n) n,     n+1,     n+1,     n+2
#   define B4(n) B2(n), B2(n+1), B2(n+1), B2(n+2)
#   define B6(n) B4(n), B4(n+1), B4(n+1), B4(n+2)
  B6(0), B6(1), B6(1), B6(2)
};

int count_ones(unsigned int v) {
  unsigned char * p = (unsigned char *) &v;
  return BitsSetTable256[p[0]] +
         BitsSetTable256[p[1]] +
         BitsSetTable256[p[2]] +
         BitsSetTable256[p[3]];
}

int msb(unsigned int x) {
  union bits { double a; int b[2]; };
  union bits v;
  v.a = x;
  return (v.b[1] >> 20) - 1023;
}

int combine_bits_wrapped(unsigned int mask, unsigned int v, int* vector, int max_vector_size) {
  unsigned int i = count_ones(mask);
  unsigned int j = 0;
  unsigned int new_v = 0;
  while (mask) {
    // Go through ones in mask.
    int msb_mask = msb(mask);
    mask &= ~(1 << msb_mask);
    // Set ones to the values of the bits in v.
    i--;
    if (v & 1 << i) {
      new_v |= 1 << msb_mask;
      vector[j] = -1;
    } else {
      vector[j] = msb_mask;
    }
    j++;
    v &= ~(1 << i);
  }

  return new_v;
}
