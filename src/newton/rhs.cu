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

static inline __device__ State rhs(const State s)
{
  const Real r2 = s.x * s.x + s.y * s.y + s.z * s.z;     // 5 FLOP
  const Real f  = -1 / ((r2 + (Real)1.0e-6) * sqrt(r2)); // 4 FLOP

  return (State){    s.u,     s.v,     s.w,
                 f * s.x, f * s.y, f * s.z}; // 3 FLOP
}

#define FLOP 12
