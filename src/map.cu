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

#ifndef DISABLE_GL

static __global__ void kernel(Point *p, const State *s, size_t n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < n) {
    const float E = 0.5f * (s[i].u * s[i].u +
                            s[i].v * s[i].v +
                            s[i].w * s[i].w);
    p[i].x = s[i].x;
    p[i].y = s[i].y;
    p[i].z = s[i].z;
    p[i].r = E;
    p[i].g = E;
    p[i].b = E;
  }
}

void map(Point *p, const State *s, size_t n)
{
  const int bsz = 256;
  const int gsz = (global::n - 1) / bsz + 1;

  kernel<<<gsz, bsz>>>(p, s, n);
}

#endif // !DISABLE_GL
