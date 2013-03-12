// Copyright (C) 2012 Chi-kwan Chan
// Copyright (C) 2012 Steward Observatory
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

#include <curand_kernel.h>

static __device__ State ic(const size_t i, const size_t n, const real t)
{
  curandStateXORWOW_t s;
  curand_init(0, i, 0, &s);

  real x, y, z, r;
  real u, v, w, V;

  do {
    x = 20 * (curand_uniform(&s) - (real)0.5);
    y = 20 * (curand_uniform(&s) - (real)0.5);
    z = 20 * (curand_uniform(&s) - (real)0.5);
    r = sqrt(x * x + y * y + z * z);
  } while(r < 2 || 10 < r);

  do {
    u = 2 * (curand_uniform(&s) - (real)0.5);
    v = 2 * (curand_uniform(&s) - (real)0.5);
    w = 2 * (curand_uniform(&s) - (real)0.5);
    V = sqrt(u * u + v * v + w * w);
  } while(1 < V);

  V  = ((real)1.5 * curand_uniform(&s) + (real)0.5) / (V * sqrt(r));
  u *= V;
  v *= V;
  w *= V;

  V  = (x * u + y * v + z * w) / (r * r);
  u -= x * V;
  v -= y * V;
  w -= z * V;

  return (State){x, y, z, u, v, w};
}
