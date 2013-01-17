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

typedef struct {
  State s;
  real  t;
} Var;

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

  cudaEventCreate(&time0);
  cudaEventCreate(&time1);
}

static void cleanup(void)
{
  if(count) {
    cudaFree(count);
    count = NULL;
  }

  cudaEventDestroy(time1);
  cudaEventDestroy(time0);
}

float evolve(Data &data, double dt)
{
  const double t = (double)data;
  const size_t n = (size_t)data;

  if(!count && !atexit(cleanup)) setup(n);

  cudaEventRecord(time0, 0);
  {
    const int bsz = 256;
    const int gsz = (n - 1) / bsz + 1;

    State *s = data.device();
    driver<<<gsz, bsz>>>(s, n, t, dt, count);
    data.deactivate();

    data += dt;
  }
  cudaEventRecord(time1, 0);

  float ms;
  cudaEventSynchronize(time1);
  cudaEventElapsedTime(&ms, time0, time1);

  size_t temp[n];
  cudaMemcpy(temp, count, sizeof(size_t) * n, cudaMemcpyDeviceToHost);

  double sum = 0, max = 0;
  for(size_t i = 0; i < n; ++i) {
    double x = temp[i];
    sum += x;
    max  = max > x ? max : x;
  }

  print("t = %6.2f, %.0f ms/%.0f steps, %6.2f Gflops, slow down by %f\n",
        (double)data, ms, sum, 1e-6 * flop() * sum / ms, n * max / sum);

  return ms;
}
