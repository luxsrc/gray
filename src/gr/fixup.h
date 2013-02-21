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

static inline __device__ int fixup(State &y, const State &s,
                                             const State &k, real dt)
{
  size_t count = 0;

  if(y.theta > (real)M_PI || y.theta < 0) {
    #pragma unroll
    EACH(y) = GET(s) + 2 * dt * GET(k); // jump over the pole by forward Euler
    ++count;
  }

  if(y.theta > (real)M_PI) {
    y.phi   += (real)M_PI;
    y.theta  = -y.theta + 2 * (real)M_PI;
    y.ktheta = -y.ktheta;
    ++count;
  }
  if(y.theta < 0) {
    y.phi   -= (real)M_PI;
    y.theta  = -y.theta;
    y.ktheta = -y.ktheta;
    ++count;
  }

  return count;
}
