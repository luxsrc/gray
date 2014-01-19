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

#define FLOP_RHS 84
#define R_SCHW   2

#define CONST_c     ((real)2.99792458e+10)
#define CONST_h     ((real)6.62606957e-27)
#define CONST_G     ((real)6.67384800e-08)
#define CONST_kB    ((real)1.38064881e-16)
#define CONST_e     ((real)4.80320425e-10)
#define CONST_me    ((real)9.10938291e-28)
#define CONST_mp_me ((real)1836.152672450)
#define CONST_mSun  ((real)1.98910000e+33)

#define T_MIN  ((real)1e-1)
#define T_MAX  ((real)1e+2)
#define T_GRID (60)

static __device__ __constant__ real log_K2it_tab[] = {
  -10.747001122, -9.5813378172, -8.5317093904, -7.5850496322, -6.7296803564,
  -5.9551606678, -5.2521532618, -4.6123059955, -4.0281471473, -3.4929929282,
  -3.0008659288, -2.5464232845, -2.1248934192, -1.7320202979, -1.3640141782,
  -1.0175079137, -0.6895179334, -0.3774091024, -0.0788627660, +0.2081526098,
  +0.4854086716, +0.7544426322, +1.0165811787, +1.2729629642, +1.5245597366,
  +1.7721960959, +2.0165678441, +2.2582588804, +2.4977566043, +2.7354658112,
  +2.9717210921, +3.2067977811, +3.4409215189, +3.6742765257, +3.9070126886,
  +4.1392515843, +4.3710915520, +4.6026119396, +4.8338766306, +5.0649369599,
  +5.2958341090, +5.5266010659, +5.7572642218, +5.9878446670, +6.2183592400,
  +6.4488213736, +6.6792417767, +6.9096289812, +7.1399897815, +7.3703295860,
  +7.6006526984, +7.8309625420, +8.0612618396, +8.2915527560, +8.5218370124,
  +8.7521159766, +8.9823907360, +9.2126621546, +9.4429309191, +9.6731975749,
  +9.9034625556
};

static inline __device__ real K2it(real te)
{
  const real h = T_GRID * log(te / T_MIN) / log(T_MAX / T_MIN);
  const int  i = h;
  const real d = h - i;

  return exp((1 - d) * log_K2it_tab[i] + d * log_K2it_tab[i+1]);
}

static inline __device__ real B_Planck(real nu, real te)
{
  nu /= CONST_c;

  return 2 * CONST_h * CONST_c * nu * nu * nu /
    (exp(CONST_h / (CONST_me * CONST_c) * (nu / te)) - 1);
}

static inline __device__ real j_synchr(real nu, real te, real ne,
                                       real B,  real cos_theta)
{
  const real nus = CONST_e / (9 * (real)M_PI * CONST_me * CONST_c) *
    te * te * B * sqrt(1 - cos_theta * cos_theta);
  const real x   = nu / nus;

  if(te        <= T_MIN ||
     cos_theta <=    -1 ||
     cos_theta >=     1 ||
     x         <=     0) return 0;

  const real cbrtx = cbrt(x);
  const real xx    = sqrt(x) + (real)1.88774862536 * sqrt(cbrtx);
  const real K2    = (te > T_MAX) ? 2 * te * te - (real)0.5 : K2it(te);

  return (real)M_SQRT2 * (real)M_PI * CONST_e * CONST_e / (3 * CONST_c) *
    ne * nus * xx * xx * exp(-cbrtx) / K2;
}

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
    kt     = -(g33 + s.bimpact * g30) * tmp; // assume E = -k_t = 1, see ic.h
    kphi   =  (g30 + s.bimpact * g00) * tmp; // assume E = -k_t = 1, see ic.h
  } // 25 FLOP

  real ar, atheta;
  {
    ar = cs * R_SCHW * s.r / (g22 * g22); // use ar as tmp

    const real G222 = -a2 * cs;
    const real G200 =  a2 * ar;
    const real G230 = -a_spin * ar * (g22 + a2 * s2);
    const real G233 = -a_spin * G230 * s2 + g33_s2 * cs;

    ar = G222 / Dlt; // use ar as tmp, will be reused in the next block

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
              - 2 * ar   * s.kr     * s.ktheta // tmp ar from the atheta block
              + 2 * G130 *   kphi   *   kt    ) / g11;
  } // 24 FLOP

  real dtau = 0, dI = 0;

  if(field) { // loaded HARM data
    int h2, h3;

    int itheta = s.theta / (real)0.021594524 - (real)9.7404968;
    if(itheta < 0) itheta = 0; else if(itheta > ntheta-1) itheta = ntheta-1;
    {
      int ir = logf(s.r - (real)0.1) / (real)0.0320826 - (real)12.637962634;
      if(ir < 0) ir = 0; else if(ir > 220) ir = 220;

      int iphi = (s.phi >= 0) ?
        ((int)(nphi * s.phi / (2 * (real)M_PI) + (real)0.5) % ( nphi)):
        ((int)(nphi * s.phi / (2 * (real)M_PI) - (real)0.5) % (-nphi));
      if(iphi < 0) iphi += nphi;

      h2 = itheta * nr + ir;
      h3 = (iphi * ntheta + itheta) * nr + ir;
    }

    real ut, ur, utheta, uphi;
    real bt, br, btheta, bphi, b;

    // Construct the four vectors u^\mu and b^\mu in modified KS coordinates
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

      b = sqrt(bt     * (gKSP00 * bt     + gKSP01 * br    +
                         gKSP02 * btheta + gKSP03 * bphi) +
               br     * (gKSP01 * bt     + gKSP11 * br    +
                         gKSP12 * btheta + gKSP13 * bphi) +
               btheta * (gKSP02 * bt     + gKSP12 * br    +
                         gKSP22 * btheta + gKSP23 * bphi) +
               bphi   * (gKSP03 * bt     + gKSP13 * br    +
                         gKSP23 * btheta + gKSP33 * bphi));
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

    // Compute red shift, angle cosine between b and k, etc
    real shift, B_nu, L_j_nu;
    {
      const real k0 = -1;             // k_t
      const real k1 = g11 * s.kr;     // k_r
      const real k2 = g22 * s.ktheta; // k_theta
      const real k3 = s.bimpact;      // k_phi

      shift = -(k0 * ut + k1 * ur + k2 * utheta + k3 * uphi); // is positive
      const real bkcos =
        (k0 * bt + k1 * br + k2 * btheta + k3 * bphi) / shift / b;

      b *= CONST_c * sqrt(4 * (real)M_PI * (CONST_mp_me + 1) * CONST_me * ne_rho);
      real ne = ne_rho      * field[h3].rho;
      real te = field[h3].u / field[h3].rho * CONST_mp_me *
        ((Tp_Te + 1) / (Tp_Te + 2) / (real)1.5 + Gamma - 1) / (Tp_Te + 1) / 2;
      if(itheta < n_pole || itheta > 125 - n_pole)
	ne *= ne_pole;

      const real nu = nu0 * shift;
      B_nu   = B_Planck(nu, te);
      L_j_nu = j_synchr(nu, te, ne, b, bkcos) *
        m_BH * (CONST_G * CONST_mSun ) / (CONST_c * CONST_c); // length scale
    }

    if(L_j_nu > 0) {
      dtau = -L_j_nu * shift / B_nu;
      dI   = -L_j_nu * exp(-s.tau) / (shift * shift);
    }
  } else {
    const real dR = s.r * sin_theta - R_torus;
    if(dR * dR + r2 * c2 < 4) {
      const real shift = (1 - Omega * s.bimpact) /
        sqrt(-g00 - 2 * g30 * Omega - g33 * Omega * Omega);

      dtau = -shift;
      dI   = dtau / (exp(shift) - 1) * exp(-s.tau); // always assume nu0 == 1
    }
  } // 5 FLOP if outside torus; 31 FLOP if inside torus

  return (State){kt, s.kr, s.ktheta, kphi, ar, atheta, // null geodesic
                 0, 0, 0,                              // constants of motion
                 dI, 0, 0, 0, dtau};                   // radiative transfer
}
