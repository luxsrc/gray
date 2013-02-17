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

static inline __device__ int fixup(State &s)
{
  const real pi = M_PI;

  if(s.theta > pi) {
    s.phi   += pi;
    s.theta  = -s.theta + 2 * pi;
    s.ktheta = -s.ktheta;
    return 1;
  }

  if(s.theta < 0) {
    s.phi   -= pi;
    s.theta  = -s.theta;
    s.ktheta = -s.ktheta;
    return 1;
  }

  return 0;
}
