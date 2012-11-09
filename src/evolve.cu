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

static __global__ void kernel(State *s, size_t n, Real dt)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < n) for(int h = 0; h < 100; ++h) {
    const Real x = s[i].x;
    const Real y = s[i].y;
    const Real z = s[i].z;
    const Real r = sqrt(x * x + y * y + z * z); // 6 FLOP
    const Real f = dt / (r * r * r);            // 3 FLOP

    s[i].x += dt * (s[i].u -= f * x); // 4 FLOP
    s[i].y += dt * (s[i].v -= f * y); // 4 FLOP
    s[i].z += dt * (s[i].w -= f * z); // 4 FLOP
  }
}

void evolve(void)
{
  const int bsz = 256;
  const int gsz = (global::n - 1) / bsz + 1;

  kernel<<<gsz, bsz>>>(global::s, global::n, 1.0e-3);
}
