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

#include "geode.hpp"
#include <iomanip>

typedef struct {
  State s;
  Real  t;
} Var;

#define GET(s)  ((Real *)&(s))[index]
#define EACH(s) for(int index = 0; index < NVAR; ++index) GET(s)

#include <rhs.cu>
#include <getdt.cu>
#include <rk4.cu>

static __global__ void kernel(State *s, size_t n, Real t, Real dt)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < n) {
    Var v = {s[i], t};
    t += dt; // make t the final time

    if(dt > 0)
      while(v.t < t)
        v = scheme(v, t - v.t);
    else
      while(v.t > t)
        v = scheme(v, t - v.t);

    s[i] = v.s;
  }
}

void evolve(double dt)
{
  using namespace global;

  cudaEventRecord(c0, 0);
  {
    const int bsz = 256;
    const int gsz = (n - 1) / bsz + 1;

    kernel<<<gsz, bsz>>>(s, n, t, dt);
    t += dt;
  }
  cudaEventRecord(c1, 0);

  float ms;
  cudaEventSynchronize(c1);
  cudaEventElapsedTime(&ms, c0, c1);

  using namespace std;

  cout
    << fixed << setprecision(2)
    << "t = " << setw(6) << t << ", "
    << ms                     << " ms/step, "
    << 1e-6 * flop() * n / ms << " Gflops"
    << std::endl;
}
