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

static inline __device__ real getdt(const State &s, real t,
                                    const State &a, real dt_max)
{
  const real r_bh = 1 + sqrt(1 - c.a_spin * c.a_spin);
  const real cap  = s.r * (1 - pow(c.r[0]/c.r[c.n_r-1], (real)0.125/c.n_r));

  if(s.r < r_bh + c.epsilon)
    return 0; // too close to the black hole

#ifndef ENABLE_GL
  if((real)0.999 * s.r * s.r > c.r_obs * c.r_obs + c.imgsz * c.imgsz / 2)
    return 0; // too far away from the black hole
#endif

  if(dt_max <   0) dt_max = -dt_max;
  if(dt_max > cap) dt_max = cap;

  if(c.field) { // if we are computing images from HARM data...
    bool done = true;

    for(int i = 0; i < c.n_nu; ++i)
      if(s.tau[i] < (real)6.90775527898)
        done = false;

    if(done)
      return 0; // integration no longer affect the intensity
  }

  return min(c.dt_scale / (fabs(a.r / s.r) + fabs(a.theta) + fabs(a.phi)),
             min(fabs((s.r - r_bh) / a.r / 2),
                 dt_max));
}
