/*
 * Copyright (C) 2016 Chi-kwan Chan
 * Copyright (C) 2016 Steward Observatory
 *
 * This file is part of GRay2.
 *
 * GRay2 is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GRay2 is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
 * License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GRay2.  If not, see <http://www.gnu.org/licenses/>.
 */

/** \file
 ** Radiative transfer
 **
 ** Radiative transfer related functions such as the emission and
 ** extinction (absorption) coefficients.
 **/

#define CONST_c     K(2.99792458e+10)
#define CONST_h     K(6.62606957e-27)
#define CONST_G     K(6.67384800e-08)
#define CONST_kB    K(1.38064881e-16)
#define CONST_Ry    K(2.17987197e-11)
#define CONST_e     K(4.80320425e-10)
#define CONST_me    K(9.10938291e-28)
#define CONST_mp_me K(1836.152672450)
#define CONST_mSun  K(1.98910000e+33)

#define M_PI    K(3.14159265358979323846)
#define M_SQRT2 K(1.41421356237309504880)

#define T_MIN K(1e-1)
#define T_MAX K(1e+2)
#define T_GRID (60)

#define LOG(x)    log(x)    /* \todo Select the right precision for log()  */
#define SQRT(x)   sqrt(x)   /* \todo Select the right precision for sqrt() */
#define CBRT(x)   cbrt(x)   /* \todo Select the right precision for cbrt() */
#define POW(x, y) pow(x, y) /* \todo Select the right precision for pow()  */
#define EXP(x)    exp(x)    /* \todo Select the right precision for exp()  */

static __constant real log_K2it_tab[] = {
	-10.747001122, -9.5813378172, -8.5317093904, -7.5850496322,
	-6.7296803564, -5.9551606678, -5.2521532618, -4.6123059955,
	-4.0281471473, -3.4929929282, -3.0008659288, -2.5464232845,
	-2.1248934192, -1.7320202979, -1.3640141782, -1.0175079137,
	-0.6895179334, -0.3774091024, -0.0788627660, +0.2081526098,
	+0.4854086716, +0.7544426322, +1.0165811787, +1.2729629642,
	+1.5245597366, +1.7721960959, +2.0165678441, +2.2582588804,
	+2.4977566043, +2.7354658112, +2.9717210921, +3.2067977811,
	+3.4409215189, +3.6742765257, +3.9070126886, +4.1392515843,
	+4.3710915520, +4.6026119396, +4.8338766306, +5.0649369599,
	+5.2958341090, +5.5266010659, +5.7572642218, +5.9878446670,
	+6.2183592400, +6.4488213736, +6.6792417767, +6.9096289812,
	+7.1399897815, +7.3703295860, +7.6006526984, +7.8309625420,
	+8.0612618396, +8.2915527560, +8.5218370124, +8.7521159766,
	+8.9823907360, +9.2126621546, +9.4429309191, +9.6731975749,
	+9.9034625556
};

static inline real
log_K2it(real te)
{
	const real h = LOG(te/(real)T_MIN) * (real)(T_GRID / LOG(T_MAX/T_MIN));
	const int  i = h;
	const real d = h - i;

	return (1 - d) * log_K2it_tab[i] + d * log_K2it_tab[i+1];
} /* 7 FLOP */

static inline real
B_Planck(real nu, real te)
{
	real f1 = 2 * CONST_h * CONST_c;          /* ~ 4e-16 */
	real f2 = CONST_h / (CONST_me * CONST_c); /* ~ 2e-10 */

	nu /= (real)CONST_c;             /* 1e-02 -- 1e+12 */
	f1 *= nu * nu;                   /* 4e-20 -- 4e+08 */
	f2 *= nu / (te + (real)EPSILON); /* 1e-12 -- 1e+02 */

	return nu * (f2 > (real)1e-5 ?
	             f1 / (EXP(f2) - 1) :
	             (f1 / f2) / (1 + f2 / 2 + f2 * f2 / 6));
} /* 10+ FLOP */

static inline real
Gaunt(real x, real y)
{
	const real sqrt_x = SQRT(x);
	const real sqrt_y = SQRT(y);

	if(x > 1)
		return y > 1 ?
			(real)SQRT(K(3.0) / M_PI) / sqrt_y :
			(real)(SQRT(K(3.0)) / M_PI) *
			((real)LOG(K(4.0) / K(1.78107241799)) - LOG(y + (real)EPSILON));
	else if(x * y > 1)
		return (real)SQRT(K(12.0)) / (sqrt_x * sqrt_y);
	else if(y > sqrt_x)
		return 1;
	else {
		/* The "small-angle classical region" formulae in
		   Rybicki & Lightman (1979) and Novikov & Thorne
		   (1973) are inconsistent; it seems that both
		   versions contain typos.  TODO: double-check the
		   following formula */
		const real g = (real)(SQRT(K(3.0)) / M_PI) *
			((real)LOG(K(4.0) / POW(K(1.78107241799), K(2.5))) + LOG(sqrt_x / (y + (real)EPSILON)));
		return g > (real)EPSILON ? g : (real)EPSILON;
	}
} /* 3+ FLOP */

static inline real
L_j_ff(real nu, real te, real ne)
{
	/* "Standard" formula for thermal bremsstrahlung, Rybicki &
	   Lightman equation (5.14b) divided by 4 pi.
	   Because the physical length scale L has to be part of the
	   radiative transfer, we multiple it with the emissivity
	   j_ff. */

	/* Assume Z == 1 and ni == ne */

	real x = CONST_me * CONST_c * CONST_c / CONST_Ry;  /* ~ 4e4 */
	real y = CONST_h / (CONST_me * CONST_c * CONST_c); /* ~ 3e-21 */
	real f = SQRT(CONST_G * CONST_mSun / (CONST_c * CONST_c) * K(6.8e-38) /
	              (4 * M_PI * SQRT(CONST_me * CONST_c * CONST_c / CONST_kB)));

	x *= te;      /* ~ 1e+04 */
	y *= nu / te; /* ~ 1e-10 */
	f *= ne;      /* ~ 1e-15 */

	return (M_ADM * f * Gaunt(x, y)) * (f / (SQRT(te) * EXP(y) + (real)EPSILON));
} /* 12 FLOP + FLOP(Gaunt) == 15+ FLOP */

static inline real
L_j_syn(real nu, real te, real ne, real B,  real cos_theta)
{
	/* An approximate expression for thermal magnetobremsstrahlung
	   emission, see Leung, Gammie, & Noble (2011) equation (72).
	   Because the physical length scale L has to be part of the
	   radiative transfer, we multiple it with the emissivity j_ff. */

	if(te        <= (real)T_MIN ||
	   cos_theta <=          -1 ||
	   cos_theta >=           1) return 0;

	const real nus = te * te * B * SQRT(1 - cos_theta * cos_theta) *
		         (real)(CONST_e / (9 * M_PI * CONST_me * CONST_c)); /* ~ 1e5 */
	const real x   = nu / (nus + (real)EPSILON); /* 1e6 -- 1e18 */

	const real f      = (CONST_G * CONST_mSun / (CONST_c * CONST_c)) *
		            (M_SQRT2 * M_PI * CONST_e * CONST_e / (3 * CONST_c));
	const real cbrtx  = CBRT(x);                                     /* 1e2 -- 1e6 */
	const real xx     = SQRT(x) + (real)1.88774862536 * SQRT(cbrtx); /* 1e3 -- 1e9 */
	const real log_K2 = (te > (real)T_MAX) ?
		            LOG(2 * te * te - (real)0.5) :
		            log_K2it(te);

	return (M_ADM * xx * EXP(-cbrtx)) * (xx * EXP(-log_K2)) * (f * ne * nus);
} /* 25 FLOP + min(4 FLOP, FLOP(log_K2it)) == 29+ FLOP */



struct rt {
	real I  [n_freq];
	real tau[n_freq];
};

struct rt
rt_icond(void)
{
	return (struct rt){{0}};
}

struct rt
rt_rhs(struct rt r, struct flow f)
{
	for(whole i = 0; i < n_freq; ++i) {
		const real nu     = nus[i] * f.shift;
		const real B_nu   = B_Planck(nu, f.te);
		const real L_j_nu = L_j_syn(nu, f.te, f.ne, f.b, f.bkcos) + L_j_ff(nu, f.te, f.ne);

		r.I  [i] = -L_j_nu * EXP(-r.tau[i]) / (f.shift * f.shift + (real)EPSILON);
		r.tau[i] = -L_j_nu * f.shift        / (B_nu              + (real)EPSILON);
	}

	return r;
}
