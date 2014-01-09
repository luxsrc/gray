// Copyright (C) 2012,2013 Chi-kwan Chan
// Copyright (C) 2012,2013 Steward Observatory
//
// This file is part of GRay.
//
// GRay is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// GRay is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
// or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
// License for more details.
//
// You should have received a copy of the GNU General Public License
// along with GRay.  If not, see <http://www.gnu.org/licenses/>.

#include "gray.h"
#include <cstdlib>
#include <para.h>
#include <rhs.h>
#include <getdt.h>

#define GET(s)  ((real *)&(s))[index]
#define EACH(s) for(int index = 0; index < NVAR; ++index) GET(s)
#  include <fixup.h>
#  include "scheme.h"
#undef GET
#undef EACH

#ifdef PARTICLE_TIME
#  define GET_TIME (t = shared[threadIdx.x].PARTICLE_TIME)
#else
#  define GET_TIME t
#endif
#  include "driver.h"
#undef GET_TIME

static size_t *count = NULL;
static size_t *temp  = NULL;
static cudaEvent_t time0, time1;

static void setup(size_t n)
{
  if(cudaSuccess != cudaMalloc((void **)&count, sizeof(size_t) * n))
    error("evolve(): fail to allocate device memory\n");

  if(NULL == (temp = (size_t *)malloc(sizeof(size_t) * n)))
    error("evolve(): fail to allocate host memory\n");

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

  if(temp) {
    free(temp);
    temp = NULL;
  }

  if(cudaSuccess != cudaEventDestroy(time1) ||
     cudaSuccess != cudaEventDestroy(time0))
    error("evolve(): fail to destroy timer\n");
}

double evolve(Data &data, double dt)
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
  cudaError_t err = cudaDeviceSynchronize();
  data.deactivate();

  if(cudaSuccess != cudaEventRecord(time1, 0))
    error("evolve(): fail to record event\n");
  if(cudaSuccess != err)
    error("evolve(): fail to launch kernel [%s]\n",
          cudaGetErrorString(err));

  float ms;
  if(cudaSuccess != cudaEventSynchronize(time1) ||
     cudaSuccess != cudaEventElapsedTime(&ms, time0, time1))
    error("evolve(): fail to obtain elapsed time\n");

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

  if(actual) {
    print("t =%7.2f; %.0f ms/%.0f steps ~%7.2f Gflops (%.2f%%),%7.2fGB/s\n",
          global::t, ms, actual, 1e-6 * flop() * actual / ms,
          100 * actual / peak,   1e-6 * rwsz() * n      / ms); // read + write
    return ms;
  } else
    return 0;
}

bool prob_config(const char *arg)
{
  debug("prob_config(""%s"")\n", arg);

  return config(arg[0], atof(arg + 2));
}
