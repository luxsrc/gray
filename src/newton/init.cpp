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

#include <math.h>

static inline State init(int i)
{
  Real x, y, z, R;
  Real u, v, w, V;

  do {
    x = 20.0 * ((double)rand() / RAND_MAX - 0.5);
    y = 20.0 * ((double)rand() / RAND_MAX - 0.5);
    z = 20.0 * ((double)rand() / RAND_MAX - 0.5);
    R = sqrt(x * x + y * y + z * z);
  } while(R < 2.0 || 10.0 < R);

  do {
    u = 2.0 * ((double)rand() / RAND_MAX - 0.5);
    v = 2.0 * ((double)rand() / RAND_MAX - 0.5);
    w = 2.0 * ((double)rand() / RAND_MAX - 0.5);
    V = sqrt(u * u + v * v + w * w);
  } while(1.0 < V);

  V  = (x * u + y * v + z * w) / (R * R);
  u -= x * V;
  v -= y * V;
  w -= z * V;

  V  = (1.5 * rand() / RAND_MAX + 0.5) / sqrt((u * u + v * v + w * w) * R);
  u *= V;
  v *= V;
  w *= V;

  return (State){x, y, z, u, v, w};
}
