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

#define CONST_c     (2.99792458e+10)
#define CONST_h     (6.62606957e-27)
#define CONST_G     (6.67384800e-08)
#define CONST_kB    (1.38064881e-16)
#define CONST_e     (4.80320425e-10)
#define CONST_me    (9.10938291e-28)
#define CONST_mp_me (1836.152672450)
#define CONST_mSun  (1.98910000e+33)

#define T_MIN  0.3
#define T_MAX  100
#define T_GRID 200

static __device__ __constant__ real log_K2it_tab[] = {
  -3.22106608, -3.09865291, -2.97870215, -2.86113209, -2.74586332,
  -2.63281867, -2.52192316, -2.41310394, -2.30629020, -2.20141320,
  -2.09840612, -1.99720410, -1.89774413, -1.79996503, -1.70380742,
  -1.60921365, -1.51612775, -1.42449542, -1.33426399, -1.24538233,
  -1.15780089, -1.07147159, -0.98634783, -0.90238445, -0.81953767,
  -0.73776508, -0.65702562, -0.57727948, -0.49848818, -0.42061443,
  -0.34362218, -0.26747652, -0.19214373, -0.11759118, -0.04378736,
   0.02929820,  0.10169491,  0.17343118,  0.24453444,  0.31503113,
   0.38494679,  0.45430602,  0.52313255,  0.59144924,  0.65927811,
   0.72664038,  0.79355646,  0.86004599,  0.92612786,  0.99182026,
   1.05714064,  1.12210578,  1.18673182,  1.25103422,  1.31502785,
   1.37872695,  1.44214521,  1.50529572,  1.56819107,  1.63084327,
   1.69326387,  1.75546389,  1.81745391,  1.87924404,  1.94084394,
   2.00226285,  2.06350961,  2.12459266,  2.18552007,  2.24629952,
   2.30693837,  2.36744364,  2.42782200,  2.48807985,  2.54822325,
   2.60825800,  2.66818964,  2.72802341,  2.78776432,  2.84741715,
   2.90698644,  2.96647649,  3.02589142,  3.08523513,  3.14451134,
   3.20372357,  3.26287519,  3.32196937,  3.38100915,  3.43999740,
   3.49893685,  3.55783008,  3.61667957,  3.67548764,  3.73425650,
   3.79298826,  3.85168490,  3.91034832,  3.96898030,  4.02758255,
   4.08615665,  4.14470415,  4.20322649,  4.26172502,  4.32020105,
   4.37865580,  4.43709043,  4.49550604,  4.55390368,  4.61228432,
   4.67064890,  4.72899830,  4.78733335,  4.84565484,  4.90396353,
   4.96226011,  5.02054526,  5.07881960,  5.13708374,  5.19533824,
   5.25358362,  5.31182041,  5.37004906,  5.42827005,  5.48648378,
   5.54469067,  5.60289110,  5.66108542,  5.71927398,  5.77745709,
   5.83563506,  5.89380818,  5.95197672,  6.01014093,  6.06830105,
   6.12645732,  6.18460995,  6.24275914,  6.30090508,  6.35904796,
   6.41718795,  6.47532521,  6.53345990,  6.59159215,  6.64972211,
   6.70784989,  6.76597564,  6.82409945,  6.88222144,  6.94034171,
   6.99846036,  7.05657747,  7.11469314,  7.17280744,  7.23092046,
   7.28903226,  7.34714291,  7.40525248,  7.46336103,  7.52146861,
   7.57957528,  7.63768109,  7.69578610,  7.75389034,  7.81199385,
   7.87009669,  7.92819888,  7.98630047,  8.04440148,  8.10250195,
   8.16060191,  8.21870140,  8.27680042,  8.33489902,  8.39299722,
   8.45109503,  8.50919249,  8.56728960,  8.62538639,  8.68348288,
   8.74157909,  8.79967502,  8.85777071,  8.91586615,  8.97396136,
   9.03205637,  9.09015117,  9.14824578,  9.20634021,  9.26443447,
   9.32252857,  9.38062252,  9.43871633,  9.49681001,  9.55490355,
   9.61299698,  9.67109030,  9.72918351,  9.78727662,  9.84536963,
   9.90346256
};

static inline __device__ real K2it(real t_e)
{
  const real h = T_GRID * log(t_e / T_MIN) / log(T_MAX / T_MIN);
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

static inline __device__ real j_synchr(real nu,
                                       real t_e, real n_e, real B,
                                       real cos_theta)
{
  if(t_e < T_MIN) return 0;

  const real nus = CONST_e / (9 * M_PI * CONST_me * CONST_c) *
    t_e * t_e * B * sqrt(1 - cos_theta * cos_theta);

  if(nu > 1e12 * nus) return 0;

  const real K2    = (t_e > T_MAX) ? 2 * t_e * t_e : K2it(t_e);
  const real x     = nu / nus;
  const real cbrtx = cbrt(x);
  const real xx    = sqrt(x) + 1.88774862536 * sqrt(cbrtx);

  return M_SQRT2 * M_PI * CONST_e * CONST_e / (3 * CONST_c) *
    n_e * nus * xx * xx * exp(-cbrtx) / K2;
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

  real dtau = 0, df = 0;
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

    const real ne = ne_rho      * field[h3].rho;
    const real te = field[h3].u / field[h3].rho * CONST_mp_me *
      ((2 / 3) * (Tp_Te + 1) / (Tp_Te + 2) + Gamma - 1) / (Tp_Te + 1) / 2;

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

    // Compute red shift and angle cosine between b and k
    real shift, bkcos;
    {
      const real k0 = -1;             // k_t
      const real k1 = g11 * s.kr;     // k_r
      const real k2 = g22 * s.ktheta; // k_theta
      const real k3 = s.bimpact;      // k_phi

      shift = -(k0 * ut + k1 * ur + k2 * utheta + k3 * uphi); // is positive
      bkcos =  (k0 * bt + k1 * br + k2 * btheta + k3 * bphi) / shift / b;

      if(bkcos >  1) bkcos =  1;
      if(bkcos < -1) bkcos = -1;
    }
    const real nu   = nu0 * shift;
    const real B_nu = B_Planck(nu, te);
    const real j_nu = j_synchr(nu, te, ne, b, bkcos) *
      (4.3e6 * CONST_G * CONST_mSun ) / (CONST_c * CONST_c);

    dtau = -j_nu * shift / B_nu;
    df   = -j_nu * exp(-s.tau) / (shift * shift);
  } else {
    const real dR = s.r * sin_theta - R_torus;
    if(dR * dR + r2 * c2 < 4) {
      const real shift = (1 - Omega * s.bimpact) /
        sqrt(-g00 - 2 * g30 * Omega - g33 * Omega * Omega);

      dtau = -shift;
      df   = dtau / (exp(shift) - 1) * exp(-s.tau); // always assume nu0 == 1
    }
  } // 5 FLOP if outside torus; 31 FLOP if inside torus

  return (State){kt, s.kr, s.ktheta, kphi, ar, atheta, // null geodesic
                 0, 0, 0,                              // constants of motion
                 df, 0, 0, 0, dtau};                   // radiative transfer
}
