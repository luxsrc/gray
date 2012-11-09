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

static __global__ void kernel(State *s, size_t n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < n) {
    s[i].x *= (Real)1.001;
    s[i].y *= (Real)1.001;
    s[i].z *= (Real)1.001;
  }
}

void evolve(void)
{
  const int bsz = 256;
  const int gsz = (global::n - 1) / bsz + 1;

  kernel<<<gsz, bsz>>>(global::s, global::n);
}
