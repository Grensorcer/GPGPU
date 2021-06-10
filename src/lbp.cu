#include <stdint.h>
#include <iostream>

#include "lbp.hh"

[[gnu::noinline]]
void _abortError(const char* msg, const char* fname, int line)
{
  cudaError_t err = cudaGetLastError();
  std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)


struct rgba8_t {
  int8_t r;
  int8_t g;
  int8_t b;
  int8_t a;
};

__global__ void heat_lut(float x)
{
}

void pouet()
{
  heat_lut<<<1, 1>>>(3);
}

