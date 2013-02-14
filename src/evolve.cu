// Copyright (C) 2012 Chi-kwan Chan
// Copyright (C) 2012 Steward Observatory
//
// This file is part of geode.
//
// Geode is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Geode is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
// or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
// License for more details.
//
// You should have received a copy of the GNU General Public License
// along with geode.  If not, see <http://www.gnu.org/licenses/>.

#include "geode.h"

#define GET(s)  ((real *)&(s))[index]
#define EACH(s) for(int index = 0; index < NVAR; ++index) GET(s)

#include <rhs.hu>
#include <getdt.hu>
#include "scheme.hu"
#include "driver.hu"

static size_t *count = NULL;
static cudaEvent_t time0, time1;

static void setup(size_t n)
{
  if(cudaSuccess != cudaMalloc((void **)&count, sizeof(size_t) * n))
    error("evolve(): fail to allocate device memory\n");

  if(cudaSuccess != cudaEventCreate(&time0) ||
     cudaSuccess != cudaEventCreate(&time1))
    error("evolve(): fail to create timer\n");
}

static void cleanup(void)
{
  if(count) {
    cudaFree(count);
    count = NULL;
  }

  if(cudaSuccess != cudaEventDestroy(time1) ||
     cudaSuccess != cudaEventDestroy(time0))
    error("evolve(): fail to destroy timer\n");
}

float evolve(Data &data, double dt)
{
  debug("evolve(*%p, %g)\n", &data, dt);

  const double t = global::t;
  const size_t n = data;

  if(!count && !atexit(cleanup)) setup(n);

  const size_t bsz = 64;
  const size_t gsz = (n - 1) / bsz + 1;

  if(cudaSuccess != cudaEventRecord(time0, 0))
    error("evolve(): fail to record event\n");

  State *s = data.device();
  driver<<<gsz, bsz, sizeof(State) * bsz>>>(s, count, n, t, global::t += dt);
  cudaError_t err = cudaGetLastError();
  data.deactivate();

  if(cudaSuccess != cudaEventRecord(time1, 0))
    error("evolve(): fail to record event\n");
  if(cudaSuccess != err)
    error("evolve(): fail to launch kernel; %s\n", cudaGetErrorString(err));

  float ms;
  if(cudaSuccess != cudaEventSynchronize(time1) ||
     cudaSuccess != cudaEventElapsedTime(&ms, time0, time1))
    error("evolve(): fail to obtain elapsed time\n");

  size_t temp[n];
  if(cudaSuccess !=
     cudaMemcpy(temp, count, sizeof(size_t) * n, cudaMemcpyDeviceToHost))
    error("evolve(): fail to copy memory from device to host\n");

  double actual = 0, peak = 0;
  for(size_t j = 0, h = 0; j < gsz; ++j) {
    size_t sum = 0, max = 0;
    for(size_t i = 0; i < bsz; ++i, ++h) {
      const size_t x = (h < n) ? temp[h] : 0;
      sum += x;
      if(max < x) max = x;
    }
    actual += sum;
    peak   += max * bsz;
  }
  print("\
t = %6.2f; %3.0f ms/%.0f steps ~ %6.2f Gflops; occupation = %5.2f%%\n\
", global::t, ms, actual, 1e-6 * flop() * actual / ms, 100 * actual / peak);

  return ms;
}
