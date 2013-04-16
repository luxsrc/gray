// Copyright (C) 2012,2013 Chi-kwan Chan
// Copyright (C) 2012,2013 Steward Observatory
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

#define FLOP_RHS 160
#define R_SCHW   2

static inline __device__ State rhs(const State &s, real t)
{
  const real a2 = a_spin * a_spin;
  const real r2 = s.r * s.r; // 1 FLOP

  real sin_theta, cos_theta, c2, cs, s2;
  {
    sincos(s.theta, &sin_theta, &cos_theta);

    c2 = cos_theta * cos_theta;
    cs = cos_theta * sin_theta;
    s2 = sin_theta * sin_theta;
  } // 4 FLOP

  real g00, g11, g22, g33, g30, g33_s2, Dlt, kt, kphi;
  {
    real sum, tmp;
    sum    = r2 + a2;
    Dlt    = sum - R_SCHW * s.r;

    g22    = sum - a2 * s2; // = r2 + a2 * [cos(theta)]^2 = Sigma
    g11    = g22 / Dlt;

    tmp    = R_SCHW * s.r / g22;
    g00    = tmp - 1;
    g30    = -a_spin * tmp * s2;
    g33_s2 = sum - a_spin * g30;
    g33    = g33_s2 * s2;

    tmp    = 1 / (g33 * g00 - g30 * g30);
    kt     = -(g33 + s.bimpact * g30) * tmp;
    kphi   =  (g30 + s.bimpact * g00) * tmp;
  } // 25 FLOP

  real ar, atheta;
  {
    ar = cs * R_SCHW * s.r / (g22 * g22); // use ar as tmp

    const real G222 = -a2 * cs;
    const real G200 =  a2 * ar;
    const real G230 = -a_spin * ar * (g22 + a2 * s2);
    const real G233 = -a_spin * G230 * s2 + g33_s2 * cs;

    ar = G222 / Dlt; // use ar as tmp, will be used in the next block

    atheta = (+     G200 *   kt     *   kt
              +     ar   * s.kr     * s.kr
              -     G222 * s.ktheta * s.ktheta
              +     G233 *   kphi   *   kphi
              - 2 * s.r  * s.kr     * s.ktheta
              + 2 * G230 *   kphi   *   kt    ) / g22;
  } // 25 FLOP

  {
    const real G111 = (s.r + (R_SCHW / 2 - s.r) * g11) / Dlt;
    const real G100 = -(R_SCHW / 2) * (r2 - a2 * c2) / (g22 * g22);
    const real G130 = -a_spin * s2 * G100;
    const real G133 = (s.r - a_spin * G130) * s2;

    ar     = (+     G100 *   kt     *   kt
              -     G111 * s.kr     * s.kr
              +     s.r  * s.ktheta * s.ktheta
              +     G133 *   kphi   *   kphi
              - 2 * ar   * s.kr     * s.ktheta // ar is from the atheta block
              + 2 * G130 *   kphi   *   kt    ) / g11;
  } // 24 FLOP

  real src_R = 0, src_G = 0, src_B = 0;
  if(field) {
    int h2, h3;
    {
      int ir = (logf((real)(s.r - 0.1)) - 0.4215) / 0.0320826 + 0.5;
      if(ir < 0) ir = 0; else if(ir > 240) ir = 240;

      int itheta = s.theta / 0.0215945 - 10.2406 + 0.5;
      if(itheta < 0) itheta = 0; else if(itheta > 125) itheta = 125;

      int iphi = (s.phi >= 0) ? ((int)(60 * s.phi / (2 * M_PI) + 0.5) %  60):
                                ((int)(60 * s.phi / (2 * M_PI) - 0.5) % -60);
      if(iphi < 0) iphi += 60;

      h2 = itheta * 264 + ir;
      h3 = (iphi * 126 + itheta) * 264 + ir;
    }

    const real rho = field[h3].rho;

    // Vector u
    real u1 = field[h3].v1;
    real u2 = field[h3].v2;
    real u3 = field[h3].v3;
    real u0 = 1 / sqrt(-(coord[h2].gcov[0][0]           +
                         coord[h2].gcov[1][1] * u1 * u1 +
                         coord[h2].gcov[2][2] * u2 * u2 +
                         coord[h2].gcov[3][3] * u3 * u3 +
                         2 * (coord[h2].gcov[0][1] * u1      +
                              coord[h2].gcov[0][2] * u2      +
                              coord[h2].gcov[0][3] * u3      +
                              coord[h2].gcov[1][2] * u1 * u2 +
                              coord[h2].gcov[1][3] * u1 * u3 +
                              coord[h2].gcov[2][3] * u2 * u3)));
    u1 *= u0;
    u2 *= u0;
    u3 *= u0;

    // Transform vector u from KSP to KS coordinates
    {
      const real tmp1 = u1, tmp2 = u2;
      const real det12 = coord[h2].dxdxp[1][1] * coord[h2].dxdxp[2][2]
                       - coord[h2].dxdxp[1][2] * coord[h2].dxdxp[2][1];
      u0 /= coord[h2].dxdxp[0][0];
      u1  = (tmp1 * coord[h2].dxdxp[2][2] -
             tmp2 * coord[h2].dxdxp[1][2]) / det12;
      u2  = (tmp2 * coord[h2].dxdxp[1][1] -
             tmp1 * coord[h2].dxdxp[1][2]) / det12;
      u3 /= coord[h2].dxdxp[3][3];
    }

    // Transform vector u from KS to BL coordinates
    u0 += u1 * R_SCHW * s.r / Dlt;
    u3 += u1 * a_spin       / Dlt;

    // Compute red shift
    const real shift =
      -(-u0 + g11 * s.kr * u1 + g22 * s.ktheta * u2 + s.bimpact * u3);

    real nu;
    nu = 4 * shift; src_R = 1000 * rho * nu / (exp(nu) - 1);
    nu = 5 * shift; src_G = 1000 * rho * nu / (exp(nu) - 1);
    nu = 6 * shift; src_B = 1000 * rho * nu / (exp(nu) - 1);
  } // 86 FLOPS
  else {
    const real dR = s.r * sin_theta - R_torus;
    if(dR * dR + r2 * c2 < 4) {
      const real shift = (1 - Omega * s.bimpact) /
        sqrt(-g00 - 2 * g30 * Omega - g33 * Omega * Omega);

      real nu;
      nu = 0.4 * shift; src_R = 10 * nu / (exp(nu) - 1);
      nu = 0.5 * shift; src_G = 10 * nu / (exp(nu) - 1);
      nu = 0.6 * shift; src_B = 10 * nu / (exp(nu) - 1);
    }
  } // 5 FLOP if outside torus; 31 FLOP if inside torus

  return (State){kt, s.kr, s.ktheta, kphi, ar, atheta, // null geodesic
                 0,     0,     0,                      // constants of motion
                -src_R,-src_G,-src_B};                 // radiative transfer
}
