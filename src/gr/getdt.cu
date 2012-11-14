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

#define DT_MIN 1.0e-2
#define STEP_FACTOR 32

static __device__ Real getdt(const Var v, const State a)
{
  Real dt = fabs(v.s.r / a.r);

  dt = min(dt, fabs(v.s.theta / a.theta));
  dt = min(dt, fabs(v.s.phi   / a.phi  ));

  return max(dt, (Real)DT_MIN) / STEP_FACTOR;
}
