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

#define R_SCHW 2

static inline State init(int i)
{
  // Photon position and momentum in spherical coordinates
  real r, theta, phi, kr, ktheta, kphi;
  {
    const real cos_obs = cos(M_PI / 180 * THETA_OBS);
    const real sin_obs = sin(M_PI / 180 * THETA_OBS);

    real x, y, z;
    {
      const real alpha = 20.0 * ((double)rand() / RAND_MAX - 0.5);
      const real beta  = 20.0 * ((double)rand() / RAND_MAX - 0.5);

      x  = R_OBS * sin_obs - beta * cos_obs ;
      y  = alpha;
      z  = R_OBS * cos_obs + beta * sin_obs;
    }
    const real R2 = x * x + y * y;

    r     = sqrt(R2 + z * z);
    theta = acos(z / r);
    phi   = atan2(y, x);

    kr     = R_OBS / r;
    ktheta = (kr * z / r - cos_obs) / sqrt(R2);
    kphi   = -sin_obs * y / R2;
  }

  // Impact parameter = L / E
  real bimpact;
  {
    real s2, g00, g11, g22, g33, g30;
    {
      real sum, tmp, r2, a2;

      tmp = sin(theta);
      s2  = tmp * tmp ;
      r2  = r * r;
      a2  = A_SPIN * A_SPIN;

      sum = r2 + a2;
      g22 = sum - a2 * s2; // = r2 + a2 * [cos(theta)]^2 = Sigma
      g11 = g22 / (sum - R_SCHW * r);

      tmp = R_SCHW * r / g22;
      g00 = tmp - 1;
      g30 = -A_SPIN * tmp * s2;
      g33 = (sum - A_SPIN * g30) * s2;
    }

    real kt;
    {
      real g30_kphi = g30 * kphi;
      real Delta    = g30_kphi * g30_kphi - g00 * (g11 * kr     * kr     +
                                                   g22 * ktheta * ktheta +
                                                   g33 * kphi   * kphi  );
      kt = -(g30_kphi + sqrt(Delta)) / g00;
    }

    bimpact = -(g33 * kphi + g30 * kt) / (g00 * kt + g30 * kphi);
  }

  return (State){0.0, r, theta, phi, kr, ktheta, bimpact};
}
