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

#define R_SCHW 2.0
#define A_SPIN 0.0

static inline void metric(Real *g, Real r, Real theta)
{
  Real s2, sum, Sigma, cross;
  {
    const Real s  = sin(theta);
    const Real r2 = r * r;
    const Real a2 = A_SPIN * A_SPIN;
    s2    = s  * s ;
    sum   = r2 + a2;
    Sigma = sum - a2 * s2;
    cross = -R_SCHW * r * A_SPIN * s2 / Sigma;
  }
  g[0] = R_SCHW * r / Sigma - 1.0;    // g_tt
  g[1] = Sigma / (sum - R_SCHW * r);  // g_rr
  g[2] = Sigma;                       // g_thetatheta
  g[3] = (sum - A_SPIN * cross) * s2; // g_phiphi
  g[4] = cross;                       // g_tphi
}
