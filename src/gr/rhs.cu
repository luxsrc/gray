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
  const Real cos_theta = cos(s.theta);

  const Real a2 = A_SPIN * A_SPIN;
  const Real c2 = cos_theta * cos_theta;
  const Real r2 = s.r * s.r;
  const Real s2 = 1.0 - c2;
  const Real cs = sqrt(c2 * s2); // 6 FLOP up to here

  Real grr, gphiphi, Sigma, Delta, kt, kphi;
  {
    Real g[5]; metric(g, s.r, s.theta);
    const Real tmp = 1.0 / (g[3] * g[0] - g[4] * g[4]);

    grr     = g[1];
    gphiphi = g[3];
    Sigma   = g[2];
    Delta   = Sigma / grr;

    kt      = -(g[3] + s.bimpact * g[4]) * tmp;
    kphi    =  (g[4] + s.bimpact * g[0]) * tmp;
  } // 30 FLOP in this block

  Real ar, atheta;
  {
    const Real G111 = (s.r + (R_SCHW / 2.0 - s.r) * grr) / Delta;
    const Real G112 = -a2 * cs / Delta;
    const Real G122 = s.r;
    const Real G100 = -(R_SCHW / 2.0) * (r2 - a2 * c2) / (Sigma * Sigma);
    const Real G130 = -A_SPIN * s2 * G100;
    const Real G133 = (s.r - A_SPIN * G130) * s2;

    ar = (+       G100 *   kt     *   kt
          -       G111 * s.kr     * s.kr
          +       G122 * s.ktheta * s.ktheta
          +       G133 *   kphi   *   kphi
          - 2.0 * G112 * s.kr     * s.ktheta
          + 2.0 * G130 *   kphi   *   kt    ) / grr;
  } // 37 FLOP

  {
    const Real G222 = -a2 * cs;
    const Real G212 = s.r;
    const Real G211 = G222 / Delta;
    const Real G200 = -G222 * R_SCHW * s.r / (Sigma * Sigma);
    const Real G230 = -G200 * (Sigma + a2 * s2) / A_SPIN;
    const Real G233 = -A_SPIN * G230 * s2 + gphiphi * cs / s2;

    atheta = (+       G200 *   kt     *   kt
              +       G211 * s.kr     * s.kr
              -       G222 * s.ktheta * s.ktheta
              +       G233 *   kphi   *   kphi
              - 2.0 * G212 * s.kr     * s.ktheta
              + 2.0 * G230 *   kphi   *   kt    ) / Sigma;
  } // 38 FLOP

  return (State){kt, s.kr, s.ktheta, kphi, ar, atheta, 0.0};
}

#define FLOP 111
