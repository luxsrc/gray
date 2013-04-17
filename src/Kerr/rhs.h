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

#define FLOP_RHS 252
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
      int ir = (logf((real)(s.r - (real)0.1)) - (real)0.4215) /
        (real)0.0320826 + (real)0.5;
      if(ir < 0) ir = 0; else if(ir > 240) ir = 240;

      int itheta = s.theta / (real)0.0215945 - (real)9.7406;
      if(itheta < 0) itheta = 0; else if(itheta > 125) itheta = 125;

      int iphi = (s.phi >= 0) ?
        ((int)(60 * s.phi / (2 * (real)M_PI) + (real)0.5) %  60):
        ((int)(60 * s.phi / (2 * (real)M_PI) - (real)0.5) % -60);
      if(iphi < 0) iphi += 60;

      h2 = itheta * 264 + ir;
      h3 = (iphi * 126 + itheta) * 264 + ir;
    }

    const real den = field[h3].rho;
    const real eng = field[h3].u;

    // Construct the four vectors u^\mu and b^\mu in modified KS coordinates
    real ut, ur, utheta, uphi, bt, br, btheta, bphi, bb;
    {
      const real gKSP00 = coord[h2].gcov[0][0];
      const real gKSP11 = coord[h2].gcov[1][1];
      const real gKSP22 = coord[h2].gcov[2][2];
      const real gKSP33 = coord[h2].gcov[3][3];
      const real gKSP01 = coord[h2].gcov[0][1];
      const real gKSP02 = coord[h2].gcov[0][2];
      const real gKSP03 = coord[h2].gcov[0][3];
      const real gKSP12 = coord[h2].gcov[1][2];
      const real gKSP13 = coord[h2].gcov[1][3];
      const real gKSP23 = coord[h2].gcov[2][3];

      // Vector u
      ur     = field[h3].v1;
      utheta = field[h3].v2;
      uphi   = field[h3].v3;
      ut     = 1 / sqrt(-(gKSP00                   +
                          gKSP11 * ur     * ur     +
                          gKSP22 * utheta * utheta +
                          gKSP33 * uphi   * uphi   +
                          2 * (gKSP01 * ur              +
                               gKSP02 * utheta          +
                               gKSP03 * uphi            +
                               gKSP12 * ur     * utheta +
                               gKSP13 * ur     * uphi   +
                               gKSP23 * utheta * uphi)));
      ur     *= ut;
      utheta *= ut;
      uphi   *= ut;

      // Vector B
      br     = field[h3].B1;
      btheta = field[h3].B2;
      bphi   = field[h3].B3;
      bt     = (br     * (gKSP01 * ut     + gKSP11 * ur    +
                          gKSP12 * utheta + gKSP13 * uphi) +
                btheta * (gKSP02 * ut     + gKSP12 * ur    +
                          gKSP22 * utheta + gKSP23 * uphi) +
                bphi   * (gKSP03 * ut     + gKSP13 * ur    +
                          gKSP23 * uphi   + gKSP33 * uphi));
      br     += bt * ur     / ut;
      btheta += bt * utheta / ut;
      bphi   += bt * uphi   / ut;

      bb =
        bt     * (gKSP00 * bt + gKSP01 * br + gKSP02 * btheta + gKSP03 * bphi)+
        br     * (gKSP01 * bt + gKSP11 * br + gKSP12 * btheta + gKSP13 * bphi)+
        btheta * (gKSP02 * bt + gKSP12 * br + gKSP22 * btheta + gKSP23 * bphi)+
        bphi   * (gKSP03 * bt + gKSP13 * br + gKSP23 * btheta + gKSP33 * bphi);
    }

    // Transform vector u and b from KSP to KS coordinates
    {
      const real dxdxp00 = coord[h2].dxdxp[0][0];
      const real dxdxp11 = coord[h2].dxdxp[1][1];
      const real dxdxp12 = coord[h2].dxdxp[1][2];
      const real dxdxp21 = coord[h2].dxdxp[2][1];
      const real dxdxp22 = coord[h2].dxdxp[2][2];
      const real dxdxp33 = coord[h2].dxdxp[3][3];

      const real det12 = dxdxp11 * dxdxp22 - dxdxp12 * dxdxp21;
      real temp1, temp2;

      temp1  = ur;
      temp2  = utheta;
      ut    /= dxdxp00;
      ur     = (temp1 * dxdxp22 - temp2 * dxdxp12) / det12;
      utheta = (temp2 * dxdxp11 - temp1 * dxdxp21) / det12;
      uphi  /= dxdxp33;

      temp1  = br;
      temp2  = btheta;
      bt    /= dxdxp00;
      br     = (temp1 * dxdxp22 - temp2 * dxdxp12) / det12;
      btheta = (temp2 * dxdxp11 - temp1 * dxdxp21) / det12;
      bphi  /= dxdxp33;
    }

    // Transform vector u and b from KS to BL coordinates
    {
      const real temp0 = R_SCHW * s.r / Dlt; // Note that s.r and Dlt are
      const real temp3 = a_spin       / Dlt; // evaluated at photon position

      ut   += ur * temp0;
      uphi += ur * temp3;

      bt   += br * temp0;
      bphi += br * temp3;
    }

    // Compute red shift and angle cosine between b and k
    real shift, bkcos;
    {
      const real k0 = -1;
      const real k1 = g11 * s.kr;
      const real k2 = g22 * s.ktheta;
      const real k3 = s.bimpact;

      shift =-(k0 * ut + k1 * ur + k2 * utheta + k3 * uphi); // is positive
      bkcos = (k0 * bt + k1 * br + k2 * btheta + k3 * bphi) / shift / sqrt(bb);

      if(bkcos >  1) bkcos =  1;
      if(bkcos < -1) bkcos = -1;
    }

    real nu;
    nu = 4 * shift; src_R = 1000 * den * nu / (exp(nu) - 1);
    nu = 5 * shift; src_G = 1000 * den * nu / (exp(nu) - 1);
    nu = 6 * shift; src_B = 1000 * den * nu / (exp(nu) - 1);
  } // 173 FLOPS
  else {
    const real dR = s.r * sin_theta - R_torus;
    if(dR * dR + r2 * c2 < 4) {
      const real shift = (1 - Omega * s.bimpact) /
        sqrt(-g00 - 2 * g30 * Omega - g33 * Omega * Omega);

      real nu;
      nu = 4 * shift; src_R = nu / (exp(nu / 10) - 1);
      nu = 5 * shift; src_G = nu / (exp(nu / 10) - 1);
      nu = 6 * shift; src_B = nu / (exp(nu / 10) - 1);
    }
  } // 5 FLOP if outside torus; 31 FLOP if inside torus

  return (State){kt, s.kr, s.ktheta, kphi, ar, atheta, // null geodesic
                 0,     0,     0,                      // constants of motion
                -src_R,-src_G,-src_B};                 // radiative transfer
}
