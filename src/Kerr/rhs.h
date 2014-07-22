// Copyright (C) 2012--2014 Chi-kwan Chan
// Copyright (C) 2012--2014 Steward Observatory
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

#define EPSILON  1e-32
#define FLOP_RHS (harm::using_harm ? (287 + harm::n_nu * 66) : 104)
#define RWSZ_RHS (harm::using_harm ? 25 : 0)
#define R_SCHW   2

#define CONST_c     (2.99792458e+10)
#define CONST_h     (6.62606957e-27)
#define CONST_G     (6.67384800e-08)
#define CONST_kB    (1.38064881e-16)
#define CONST_Ry    (2.17987197e-11)
#define CONST_e     (4.80320425e-10)
#define CONST_me    (9.10938291e-28)
#define CONST_mp_me (1836.152672450)
#define CONST_mSun  (1.98910000e+33)

#define T_MIN  (1e-1)
#define T_MAX  (1e+2)
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

static inline __device__ real log_K2it(real te)
{
  const real h = log(te / (real)T_MIN) * (real)(T_GRID / log(T_MAX / T_MIN));
  const int  i = h;
  const real d = h - i;

  return (1 - d) * log_K2it_tab[i] + d * log_K2it_tab[i+1];
} // 7 FLOP

static inline __device__ real B_Planck(real nu, real te)
{
  real f1 = 2 * CONST_h * CONST_c;          // ~ 4e-16
  real f2 = CONST_h / (CONST_me * CONST_c); // ~ 2e-10

  nu /= (real)CONST_c;             // 1e-02 -- 1e+12
  f1 *= nu * nu;                   // 4e-20 -- 4e+08
  f2 *= nu / (te + (real)EPSILON); // 1e-12 -- 1e+02

  return nu * (f2 > (real)1e-5 ?
               f1 / (exp(f2) - 1) :
               (f1 / f2) / (1 + f2 / 2 + f2 * f2 / 6));
} // 10+ FLOP

static inline __device__ real Gaunt(real x, real y)
{
  const real sqrt_x = sqrt(x);
  const real sqrt_y = sqrt(y);

  if(x > 1)
    return y > 1 ?
           (real)sqrt(3.0 / M_PI) / sqrt_y :
           (real)(sqrt(3.0) / M_PI) * ((real)log(4 / 1.78107241799) -
                                       log(y + (real)EPSILON));
  else if(x * y > 1)
    return (real)sqrt(12.0) / (sqrt_x * sqrt_y);
  else if(y > sqrt_x)
    return 1;
  else {
    // The "small-angle classical region" formulae in Rybicki &
    // Lightman (1979) and Novikov & Thorne (1973) are inconsistent;
    // it seems that both versions contain typos.
    // TODO: double-check the following formula
    const real g = (real)(sqrt(3.0) / M_PI) *
                   ((real)log(4.0 / pow(1.78107241799, 2.5)) +
                    log(sqrt_x / (y + (real)EPSILON)));
    return g > (real)EPSILON ? g : (real)EPSILON;
  }
} // 3+ FLOP

static inline __device__ real L_j_ff(real nu, real te, real ne)
{
  // Assume Z == 1 and ni == ne

  real x = CONST_me * CONST_c * CONST_c / CONST_Ry;    // ~ 4e4
  real y = (CONST_h / (CONST_me * CONST_c * CONST_c)); // ~ 3e-21
  real f = sqrt(CONST_G * CONST_mSun / (CONST_c * CONST_c) *
                6.8e-38 / sqrt(CONST_me * CONST_c * CONST_c / CONST_kB));

  x *= te;      // ~ 1e+04
  y *= nu / te; // ~ 1e-10
  f *= ne;      // ~ 1e-15

  return (c.m_BH * f * Gaunt(x, y)) * (f / (sqrt(te)*exp(y) + (real)EPSILON));
} // 12 FLOP + FLOP(Gaunt) == 15+ FLOP

static inline __device__ real L_j_synchr(real nu, real te, real ne,
                                         real B,  real cos_theta)
{
  const real nus = te * te * B * sqrt(1 - cos_theta * cos_theta) *
                   (real)(CONST_e / (9 * M_PI * CONST_me * CONST_c)); // ~ 1e5
  const real x   = nu / (nus + (real)EPSILON); // 1e6 -- 1e18

  if(te        <= (real)T_MIN ||
     cos_theta <=          -1 ||
     cos_theta >=           1 ||
     nus       <=           0 ||
     x         <=           0) return 0;

  const real f      = (CONST_G * CONST_mSun / (CONST_c * CONST_c)) *
                      (M_SQRT2 * M_PI * CONST_e * CONST_e / (3 * CONST_c));
  const real cbrtx  = cbrt(x);                                    // 1e2 -- 1e6
  const real xx     = sqrt(x) + (real)1.88774862536 * sqrt(cbrtx);// 1e3 -- 1e9
  const real log_K2 = (te > (real)T_MAX) ?
                      log(2 * te * te - (real)0.5) :
                      log_K2it(te);

  return (c.m_BH * xx * exp(-cbrtx)) * (xx * exp(-log_K2)) * (f * ne * nus);
} // 25 FLOP + min(4 FLOP, FLOP(log_K2it)) == 29+ FLOP

static inline __device__ State rhs(const State &s, real t)
{
  State d = {0};

  const real a2 = c.a_spin * c.a_spin; // 1 FLOP
  const real r2 = s.r * s.r;           // 1 FLOP

  real sin_theta, cos_theta, c2, cs, s2;
  {
    sincos(s.theta, &sin_theta, &cos_theta);

    c2 = cos_theta * cos_theta;
    cs = cos_theta * sin_theta;
    s2 = sin_theta * sin_theta;
  } // 4 FLOP

  real g00, g11, g22, g33, g30, g33_s2, Dlt;
  {
    real sum, tmp;
    sum    = r2 + a2;
    Dlt    = sum - R_SCHW * s.r;

    g22    = sum - a2 * s2; // = r2 + a2 * [cos(theta)]^2 = Sigma
    g11    = g22 / Dlt;

    tmp    = R_SCHW * s.r / g22;
    g00    = tmp - 1;
    g30    = -c.a_spin * tmp * s2;
    g33_s2 = sum - c.a_spin * g30;
    g33    = g33_s2 * s2;

    tmp    = 1 / (g33 * g00 - g30 * g30);
    d.t    = -(g33 + s.bimpact * g30) * tmp; // assume E = -k_t = 1, see ic.h
    d.phi  =  (g30 + s.bimpact * g00) * tmp; // assume E = -k_t = 1, see ic.h
  } // 26 FLOP

  {
    d.r = cs * R_SCHW * s.r / (g22 * g22); // use d.r as tmp

    const real G222 = -a2 * cs;
    const real G200 =  a2 * d.r;
    const real G230 = -c.a_spin * d.r * (g22 + a2 * s2);
    const real G233 = -c.a_spin * G230 * s2 + g33_s2 * cs;

    d.r = G222 / Dlt; // use d.r as tmp, will be reused in the next block

    d.ktheta = (+     G200 * d.t      * d.t
                +     d.r  * s.kr     * s.kr
                -     G222 * s.ktheta * s.ktheta
                +     G233 * d.phi    * d.phi
                - 2 * s.r  * s.kr     * s.ktheta
                + 2 * G230 * d.phi    * d.t     ) / g22;
  } // 37 FLOP

  {
    const real G111 = (s.r + (R_SCHW / 2 - s.r) * g11) / Dlt;
    const real G100 = -(R_SCHW / 2) * (r2 - a2 * c2) / (g22 * g22);
    const real G130 = -c.a_spin * s2 * G100;
    const real G133 = (s.r - c.a_spin * G130) * s2;

    d.kr = (+     G100 * d.t      * d.t
            -     G111 * s.kr     * s.kr
            +     s.r  * s.ktheta * s.ktheta
            +     G133 * d.phi    * d.phi
            - 2 * d.r  * s.kr     * s.ktheta // tmp d.r from the d.ktheta block
            + 2 * G130 * d.phi    * d.t     ) / g11;
  } // 35 FLOP

  d.r     = s.kr;
  d.theta = s.ktheta;
  if(!c.field || s.r < c.r[0]) return d;

  // Get indices to access HARM data
  int h2, h3;
  real f = (real)0.5, g = (real)0.5, h;
  {
    int I = c.n_r-1, i = I; // assume c.n_r > 1
    if(c.r[i] > s.r) {
      do I = i--; while(i && c.r[i] > s.r); // assume s.r >= c.r[0]
      f = (s.r - c.r[i]) / (c.r[I] - c.r[i]);
    } // else, constant extrapolate

    int J = c.n_theta-1, j = J;
    if(s.theta < c.coord[i].theta *    f  +
                 c.coord[I].theta * (1-f))
      j = J = 0; // constant extrapolation
    else {
      real theta = c.coord[j*c.n_r + i].theta *    f  +
                   c.coord[j*c.n_r + I].theta * (1-f);
      if(theta > s.theta) {
	real Theta;
        do {
          J = j--;
          Theta = theta;
          theta = c.coord[j*c.n_r + i].theta *    f  +
                  c.coord[j*c.n_r + I].theta * (1-f);
        } while(j && theta > s.theta);
        g = (s.theta - theta) / (Theta - theta);
      } // else, constant extrapolation
    }

    h  = s.phi / (real)(2*M_PI);
    h -= floor(h);
    h *= c.n_phi;
    int k = h, K = k == c.n_phi-1 ? 0 : k+1;
    h -= k;

    h2 = j * c.n_r + i;
    h3 = (k * c.n_theta + j) * c.n_r + (i < c.n_rx ? i : c.n_rx);
  } // 11+ FLOP

  // Construct the four vectors u^\mu and b^\mu in modified KS coordinates
  real ut, ur, utheta, uphi;
  real bt, br, btheta, bphi, b, ti_te;
  {
    const real gKSP00 = c.coord[h2].gcov[0][0];
    const real gKSP11 = c.coord[h2].gcov[1][1];
    const real gKSP22 = c.coord[h2].gcov[2][2];
    const real gKSP33 = c.coord[h2].gcov[3][3];
    const real gKSP01 = c.coord[h2].gcov[0][1];
    const real gKSP02 = c.coord[h2].gcov[0][2];
    const real gKSP03 = c.coord[h2].gcov[0][3];
    const real gKSP12 = c.coord[h2].gcov[1][2];
    const real gKSP13 = c.coord[h2].gcov[1][3];
    const real gKSP23 = c.coord[h2].gcov[2][3];

    ur     = c.field[h3].v1;
    utheta = c.field[h3].v2;
    uphi   = c.field[h3].v3;
    br     = c.field[h3].B1;
    btheta = c.field[h3].B2;
    bphi   = c.field[h3].B3;

    if(s.r > c.r[c.n_rx]) {
      // The flow is sub-Keplerian
      ur      = 0;
      utheta *= sqrt(c.r[c.n_rx] / (s.r + (real)EPSILON));
      uphi    = 0;
      // Zero out the magnetic field to void unrealistic synchrotron
      // radiation at large radius for the constant temperature model
      br      = 0;
      btheta  = 0;
      bphi    = 0;
    }

    // Vector u
    ut     = 1 / sqrt((real)EPSILON
                      -(gKSP00                   +
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
    bt     = (br     * (gKSP01 * ut     + gKSP11 * ur    +
                        gKSP12 * utheta + gKSP13 * uphi) +
              btheta * (gKSP02 * ut     + gKSP12 * ur    +
                        gKSP22 * utheta + gKSP23 * uphi) +
              bphi   * (gKSP03 * ut     + gKSP13 * ur    +
                        gKSP23 * uphi   + gKSP33 * uphi));
    br     = (br     + bt * ur    ) / ut;
    btheta = (btheta + bt * utheta) / ut;
    bphi   = (bphi   + bt * uphi  ) / ut;

    const real bb = (bt     * (gKSP00 * bt     + gKSP01 * br    +
                               gKSP02 * btheta + gKSP03 * bphi) +
                     br     * (gKSP01 * bt     + gKSP11 * br    +
                               gKSP12 * btheta + gKSP13 * bphi) +
                     btheta * (gKSP02 * bt     + gKSP12 * br    +
                               gKSP22 * btheta + gKSP23 * bphi) +
                     bphi   * (gKSP03 * bt     + gKSP13 * br    +
                               gKSP23 * btheta + gKSP33 * bphi));
    const real ibeta = bb / (2 * (c.Gamma-1) * c.field[h3].u + (real)EPSILON);
    ti_te = (ibeta > c.threshold) ? c.Ti_Te_f : c.Ti_Te_d;
    b = sqrt(bb);
  } // 107 FLOP

  // Construct the scalars rho and tgas
  real rho, tgas;
  {
    const real Gamma = (1 + (ti_te+1) / (ti_te+2) / (real)1.5 + c.Gamma) / 2;

    rho  = c.field[h3].rho;
    tgas = (Gamma - 1) * c.field[h3].u / (rho + (real)EPSILON);

    if(s.r > c.r[c.n_rx]) {
      const real invr = c.r[c.n_rx] / s.r;
      rho  *= invr;
      tgas *= invr;
    }
  } // 11 FLOP

  // Skip cell if tgas is above the threshold
  if(tgas > c.tgas_max) {
    for(int i = 0; i < c.n_nu; ++i)
      d.tau[i] = d.I[i] = 0;
    return d;
  }

  // Transform vector u and b from KSP to KS coordinates
  {
    const real dxdxp00 = c.coord[h2].dxdxp[0][0];
    const real dxdxp11 = c.coord[h2].dxdxp[1][1];
    const real dxdxp12 = c.coord[h2].dxdxp[1][2];
    const real dxdxp21 = c.coord[h2].dxdxp[2][1];
    const real dxdxp22 = c.coord[h2].dxdxp[2][2];
    const real dxdxp33 = c.coord[h2].dxdxp[3][3];

    real temp1, temp2;

    temp1  = ur;
    temp2  = utheta;
    ut    *= dxdxp00;
    ur     = (dxdxp11 * temp1 + dxdxp12 * temp2);
    utheta = (dxdxp21 * temp1 + dxdxp22 * temp2);
    uphi  *= dxdxp33;

    temp1  = br;
    temp2  = btheta;
    bt    *= dxdxp00;
    br     = (dxdxp11 * temp1 + dxdxp12 * temp2);
    btheta = (dxdxp21 * temp1 + dxdxp22 * temp2);
    bphi  *= dxdxp33;
  } // 16 FLOP

  // Transform vector u and b from KS to BL coordinates
  {
    const real temp0 = -R_SCHW * s.r / Dlt; // Note that s.r and Dlt are
    const real temp3 = -c.a_spin     / Dlt; // evaluated at photon position

    ut   += ur * temp0;
    uphi += ur * temp3;

    bt   += br * temp0;
    bphi += br * temp3;
  } // 13 FLOP

  // Compute red shift, angle cosine between b and k, etc
  real shift, bkcos, ne, te;
  {
    const real k0 = -1;             // k_t
    const real k1 = g11 * s.kr;     // k_r
    const real k2 = g22 * s.ktheta; // k_theta
    const real k3 = s.bimpact;      // k_phi

    shift = -(k0 * ut + k1 * ur + k2 * utheta + k3 * uphi); // is positive
    bkcos =  (k0 * bt + k1 * br + k2 * btheta + k3 * bphi) /
             (shift * b + (real)EPSILON);

    b *= sqrt(c.ne_rho) *
         (real)(CONST_c * sqrt(4 * M_PI * (CONST_mp_me + 1) * CONST_me));
    ne = c.ne_rho * rho;
    te = ti_te < 0 ? -ti_te : tgas * (real)CONST_mp_me / (ti_te+1);
  } // 25+ FLOP

  for(int i = 0; i < c.n_nu; ++i) {
    const real nu     = c.nu0[i] * shift;
    const real B_nu   =   B_Planck(nu, te);
    const real L_j_nu = L_j_synchr(nu, te, ne, b, bkcos) + L_j_ff(nu, te, ne);
    if(L_j_nu > 0) {
      d.I  [i] = -L_j_nu * exp(-s.tau[i]) / (shift * shift + (real)EPSILON);
      d.tau[i] = -L_j_nu * shift          / (B_nu          + (real)EPSILON);
    }
  } // 12 FLOP + FLOP(B_Planck) + FLOP(L_j_synchr) + FLOP(L_j_ff) == 66+ FLOP

  // Finally done!
  return d;
} // 104 FLOP if geodesic only; (287+) + n_nu * (66+) FLOP if HARM is on
