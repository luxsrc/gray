// Copyright (C) 2012--2014 Chi-kwan Chan
// Copyright (C) 2012--2014 Steward Observatory
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

#define FLOP_GETDT 12

static __device__ real getdt(const State &s, real t,
                             const State &a, real dt_max)
{
  const real r_bh = 1 + sqrt(1 - a_spin * a_spin);

  if(s.r   < r_bh + epsilon    ||
     s.r   > (real)1.2 * r_obs ||
     s.tau > (real)6.90775527898)
    return 0; // 0 stops the integration
  else
    return min(dt_scale / (fabs(a.r / s.r) + fabs(a.theta) + fabs(a.phi)),
               min(fabs((s.r - r_bh) / a.r / 2),
                   1.0));
}
