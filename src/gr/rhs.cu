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

static inline __device__ State rhs(const State s, const Real t)
{
  const Real a2 = A_SPIN * A_SPIN;
  const Real r2 = s.r * s.r; // 1 FLOP

  Real c2, cs, s2;
  {
    Real sin_theta, cos_theta; sincos(s.theta, &sin_theta, &cos_theta);

    c2 = cos_theta * cos_theta;
    cs = cos_theta * sin_theta;
    s2 = sin_theta * sin_theta;
  } // 4 FLOP

  Real g11, g22, g33_s2, Dlt, kt, kphi;
  {
    Real g00, g30, g33, sum, tmp;
    sum    = r2 + a2;
    Dlt    = sum - R_SCHW * s.r;

    g22    = sum - a2 * s2; // = r2 + a2 * [cos(theta)]^2 = Sigma
    g11    = g22 / Dlt;

    tmp    = R_SCHW * s.r / g22;
    g00    = tmp - 1;
    g30    = -A_SPIN * tmp * s2;
    g33_s2 = sum - A_SPIN * g30;
    g33    = g33_s2 * s2;

    tmp    = 1 / (g33 * g00 - g30 * g30);
    kt     = -(g33 + s.bimpact * g30) * tmp;
    kphi   =  (g30 + s.bimpact * g00) * tmp;
  } // 25 FLOP

  Real ar, atheta;
  {
    ar = cs * R_SCHW * s.r / (g22 * g22); // use ar as tmp

    const Real G222 = -a2 * cs;
    const Real G200 =  a2 * ar;
    const Real G230 = -A_SPIN * (ar * g22 + s2);
    const Real G233 = -A_SPIN * G230 * s2 + g33_s2 * cs;

    ar = G222 / Dlt; // use ar as tmp, will be used in the next block

    atheta = (+       G200 *   kt     *   kt
              +       ar   * s.kr     * s.kr
              -       G222 * s.ktheta * s.ktheta
              +       G233 *   kphi   *   kphi
              - 2.0 * s.r  * s.kr     * s.ktheta
              + 2.0 * G230 *   kphi   *   kt    ) / g22;
  } // 25 FLOP

  {
    const Real G111 = (s.r + (R_SCHW / 2 - s.r) * g11) / Dlt;
    const Real G100 = -(R_SCHW / 2) * (r2 - a2 * c2) / (g22 * g22);
    const Real G130 = -A_SPIN * s2 * G100;
    const Real G133 = (s.r - A_SPIN * G130) * s2;

    ar = (+       G100 *   kt     *   kt
          -       G111 * s.kr     * s.kr
          +       s.r  * s.ktheta * s.ktheta
          +       G133 *   kphi   *   kphi
          - 2.0 * ar   * s.kr     * s.ktheta // ar is from the atheta block
          + 2.0 * G130 *   kphi   *   kt    ) / g11;
  } // 24 FLOP

  return (State){kt, s.kr, s.ktheta, kphi, ar, atheta, 0.0};
}

#define FLOP 74
