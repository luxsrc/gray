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

__device__ // to make metric() a device function
#include <metric.cpp>

static inline __device__ State rhs(const State s, const Real t)
{
  Real g[5]; metric(g, s.r, s.theta); // 18 FLOP

  const Real tmp  = 1 / (g[3] * g[0] - g[4] * g[4]);  // 4 FLOP
  const Real kt   = -(g[3] + s.bimpact * g[4]) * tmp; // 4 FLOP
  const Real kphi =  (g[4] + s.bimpact * g[0]) * tmp; // 3 FLOP

  return (State){kt, s.kr, s.ktheta, kphi, 0, 0, 0};
}

#define FLOP 29
