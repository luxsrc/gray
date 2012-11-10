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

#include <rhs.cu>

#ifndef BSZ
#define BSZ 256
#endif

#ifndef FLOP
#define FLOP 0
#endif

static __global__ void kernel(State *s, size_t n, Real dt, size_t m)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < n) {
    const Real dt_2 = dt / 2;
    const Real dt_6 = dt / 6;

    State x = s[i];
    State y = x;

    for(int j = 0; j < m; ++j) {

      #define GET(s)  ((Real *)&(s))[index]
      #define EACH(s) for(int index = 0; index < NVAR; ++index) GET(s)

      const State k1 = rhs(y);
      #pragma unroll
      EACH(y) = GET(x) + dt_2 * GET(k1);

      const State k2 = rhs(y);
      #pragma unroll
      EACH(y) = GET(x) + dt_2 * GET(k2);

      const State k3 = rhs(y);
      #pragma unroll
      EACH(y) = GET(x) + dt   * GET(k3);

      const State k4 = rhs(y);
      #pragma unrol
      EACH(y) = GET(x) + dt_6 * (GET(k1) + 2 * (GET(k2) + GET(k3)) + GET(k4));

      #undef EACH
      #undef GET

      x = y;
    }

    s[i] = x;
  }
}

void rk4(size_t m)
{
  using namespace global;

  const int bsz = BSZ;
  const int gsz = (n - 1) / bsz + 1;

  kernel<<<gsz, bsz>>>(s, n, 1.0e-3, m);
}

double flop(void)
{
  return global::n * (12 * NVAR + 4 * FLOP);
}
