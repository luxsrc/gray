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

#include <metric.cpp>

#define R_OBS     10 // radius in GM/c^2
#define THETA_OBS 30 // theta in degrees

static inline State init(int i)
{
  // Photon position and momentum in spherical coordinates
  Real r, theta, phi, kr, ktheta, kphi;
  {
    const Real cos_obs = cos(M_PI / 180 * THETA_OBS);
    const Real sin_obs = sin(M_PI / 180 * THETA_OBS);

    Real x, y, z;
    {
      const Real alpha = 20.0 * ((double)rand() / RAND_MAX - 0.5);
      const Real beta  = 20.0 * ((double)rand() / RAND_MAX - 0.5);

      x  = R_OBS * sin_obs - beta * cos_obs ;
      y  = alpha;
      z  = R_OBS * cos_obs + beta * sin_obs;
    }
    const Real R2 = x * x + y * y;

    r     = sqrt(R2 + z * z);
    theta = acos(z / r);
    phi   = atan2(y, x);

    kr     = R_OBS / r;
    ktheta = (kr * z / r - cos_obs) / sqrt(R2);
    kphi   = -sin_obs * y / R2;
  }

  // Impact parameter = L / E
  Real bimpact;
  {
    Real g[5]; metric(g, r, theta);

    const Real gtphi_kphi = g[4] * kphi;
    const Real Delta = gtphi_kphi * gtphi_kphi
      - g[0] * (g[1] * kr     * kr     +
                g[2] * ktheta * ktheta +
                g[3] * kphi   * kphi  );
    const Real kt = -(g[4] * kphi + sqrt(Delta)) / g[0];

    bimpact = -(g[3] * kphi + g[4] * kt) / (g[0] * kt + g[4] * kphi);
  }

  return (State){0.0, r, theta, phi, kr, ktheta, bimpact};
}
