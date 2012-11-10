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

static __global__ void kernel(State *s, size_t n, Real dt)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < n) {
    Real x = s[i].x;
    Real y = s[i].y;
    Real z = s[i].z;
    Real u = s[i].u;
    Real v = s[i].v;
    Real w = s[i].w;

    for(int h = 0; h < 100; ++h) {
      const Real r = sqrt(x * x + y * y + z * z); // 6 FLOP
      const Real f = dt / (r * r * r);            // 3 FLOP

      x += dt * (u -= f * x); // 4 FLOP
      y += dt * (v -= f * y); // 4 FLOP
      z += dt * (w -= f * z); // 4 FLOP
    }

    s[i].x = x;
    s[i].y = y;
    s[i].z = z;
    s[i].u = u;
    s[i].v = v;
    s[i].w = w;
  }
}
