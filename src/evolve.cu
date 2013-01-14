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

#include <rhs.cu>
#include <getdt.cu>
#include <rk4.cu>

static __global__ void kernel(State *s, size_t n, real t, real dt, unsigned *p)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < n) {
    unsigned count = 0;

    Var v = {s[i], t};
    t += dt; // make t the final time

    if(dt > 0)
      while(v.t < t) {
        v = scheme(v, t - v.t);
        count++;
      }
    else
      while(v.t > t) {
        v = scheme(v, t - v.t);
        count++;
      }

    s[i] = v.s;
    p[i] = count;
  }
}

float evolve(double dt)
{
  using namespace global;

  cudaEventRecord(c0, 0);
  {
    const int bsz = 256;
    const int gsz = (n - 1) / bsz + 1;

    kernel<<<gsz, bsz>>>(s, n, t, dt, p);
    t += dt;
  }
  cudaEventRecord(c1, 0);

  float ms;
  cudaEventSynchronize(c1);
  cudaEventElapsedTime(&ms, c0, c1);

  unsigned profiler[n];
  cudaMemcpy(profiler, p, sizeof(unsigned) * n, cudaMemcpyDeviceToHost);

  double sum = 0, max = 0;
  for(size_t i = 0; i < n; ++i) {
    double x = profiler[i];
    sum += x;
    max  = max > x ? max : x;
  }

  print("t = %6.2f, %.0f ms/%.0f steps, %6.2f Gflops, slow down by %f\n",
        t, ms, sum, 1e-6 * flop() * sum / ms, n * max / sum);

  return ms;
}
