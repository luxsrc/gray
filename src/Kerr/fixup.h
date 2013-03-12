// Copyright (C) 2012 Chi-kwan Chan
// Copyright (C) 2012 Steward Observatory
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

static inline __device__ real xi(const State &s)
{
  real g00, g11, g22, g33, g30, kt, kphi;
  {
    real tmp, s2, r2, a2, sum;

    tmp  = sin(s.theta);
    s2   = tmp * tmp ;
    r2   = s.r * s.r;
    a2   = a_spin * a_spin;

    sum  = r2 + a2;
    g22  = sum - a2 * s2; // = r2 + a2 * [cos(theta)]^2 = Sigma
    g11  = g22 / (sum - R_SCHW * s.r);

    tmp  = R_SCHW * s.r / g22;
    g00  = tmp - 1;
    g30  = -a_spin * tmp * s2;
    g33  = (sum - a_spin * g30) * s2;

    tmp  = 1 / (g33 * g00 - g30 * g30);
    kt   = -(g33 + s.bimpact * g30) * tmp;
    kphi =  (g30 + s.bimpact * g00) * tmp;
  }

  return fabs(1 + (g11 * s.kr     * s.kr     +
                   g22 * s.ktheta * s.ktheta +
                   g33 *   kphi   *   kphi   +
                   g30 *   kt     *   kphi   * 2) / (g00 * kt * kt));
}

static inline __device__ real fixup(State &y, const State &s,
                                              const State &k, real dt)
{
  if(tolerance < xi(y)) {
    dt /= 9;
    #pragma unroll
    EACH(y) = GET(s) + dt * GET(k); // fall back to forward Euler
  }

  if(y.theta > (real)M_PI) {
    y.phi   += (real)M_PI;
    y.theta  = -y.theta + 2 * (real)M_PI;
    y.ktheta = -y.ktheta;
  }
  if(y.theta < 0) {
    y.phi   -= (real)M_PI;
    y.theta  = -y.theta;
    y.ktheta = -y.ktheta;
  }

  return dt;
}
