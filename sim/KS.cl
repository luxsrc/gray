
#define compute_eps_hor(index)                                          \
  if (ah_valid_##index[snap_number] > 0){                               \
                                                                        \
    /* Perform time interpolation for radius and centroid */            \
                                                                        \
    /* Note that we don't allow runs where t_final is outside the range of \
     * snapshots, so we always have a "+1" snapshot. When we are working with \
     * only one snapshot, we copied over the horizons to have two, so that \
     * we can still run through these routines. */                      \
                                                                        \
    /* We do not want to include the time in the distance, so we set    \
     * it to be the same as out point. */                               \
    centroid.s0 = q.s0;                                                 \
                                                                        \
    /* We need to do time interpolation only if the times are different \
     * Otherwise, we can just read the value */                         \
                                                                        \
    if (ah_centr_##index[snap_number].s0                                \
        != ah_centr_##index[snap_number + 1].s0) {                      \
                                                                        \
    centroid.s1 = time_interpolate(                                     \
      q.s0,                                                             \
      ah_centr_##index[snap_number].s0,       /* t1 */                  \
      ah_centr_##index[snap_number + 1].s0,   /* t2 */                  \
      ah_centr_##index[snap_number].s1,       /* y1 */                  \
      ah_centr_##index[snap_number + 1].s1);  /* y2 */                  \
                                                                        \
    centroid.s2 = time_interpolate(                                     \
      q.s0,                                                             \
      ah_centr_##index[snap_number].s0,       /* t1 */                  \
      ah_centr_##index[snap_number + 1].s0,   /* t2 */                  \
      ah_centr_##index[snap_number].s2,       /* y1 */                  \
      ah_centr_##index[snap_number + 1].s2);  /* y2 */                  \
                                                                        \
    centroid.s3 = time_interpolate(                                     \
      q.s0,                                                             \
      ah_centr_##index[snap_number].s0,       /* t1 */                  \
      ah_centr_##index[snap_number + 1].s0,   /* t2 */                  \
      ah_centr_##index[snap_number].s3,       /* y1 */                  \
      ah_centr_##index[snap_number + 1].s3);  /* y2 */                  \
                                                                        \
    radius_max = time_interpolate(                                      \
      q.s0,                                                             \
      ah_centr_##index[snap_number].s0,       /* t1 */                  \
      ah_centr_##index[snap_number + 1].s0,   /* t2 */                  \
      ah_max_r_##index[snap_number],          /* y1 */                  \
      ah_max_r_##index[snap_number + 1]);     /* y2 */                  \
    }else{                                                              \
      centroid.s1 = ah_centr_##index[snap_number].s1;                   \
      centroid.s2 = ah_centr_##index[snap_number].s2;                   \
      centroid.s3 = ah_centr_##index[snap_number].s3;                   \
      radius_max = ah_max_r_##index[snap_number];                       \
    }                                                                   \
                                                                        \
    eps_tmp = sqrt(distance(q, centroid)) - radius_max;                 \
    if (eps_tmp < eps) eps = eps_tmp;                                   \
}

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
 ** Cartesian Kerr-Schild coordinate specific schemes
 **
 ** Implement the coordinate specific functions getuu(), icond(), and
 ** rhs() in the Cartesian form of the Kerr-Schild coordiantes.  Let
 ** \f$t\f$, \f$x\f$, \f$y\f$, \f$z\f$ be the coordinates, the
 ** Cartesian Kerr-Schild metric is given by
 ** \f[
 **   g_{\mu\nu} = \gamma_{\mu\nu} + f l_\mu l_\nu
 ** \f]
 ** where \f$\gamma_{\mu\nu}\f$ is the Minkowski metric, \f$f\f$ and
 ** \f$l_\mu\f$ are defined by
 ** \f[
 **   f = \frac{2r^3}{r^4 + a^2 z^2} \mbox{ and }
 **   l_\mu = \left(1, \frac{rx + ay}{r^2 + a^2},
 **                    \frac{ry - ax}{r^2 + a^2},
 **                    \frac{z}{r}\right),
 ** \f]
 ** respectively, and \f$r\f$ is defined implicitly by\f$ x^2 + y^2 +
 ** z^2 = r^2 + a^2 (1 - z^2 / r^2)\f$.
 **/

#define a_spin 0.6

struct gr {
	real4 q;
	real4 u;
};

real
getrr(real4 q)
{
	real aa = a_spin * a_spin;
	real zz = q.s3 * q.s3;
	real kk = K(0.5) * (q.s1 * q.s1 + q.s2 * q.s2 + zz - aa);
	return sqrt(kk * kk + aa * zz) + kk;
}

real
geteps(real4 q, const whole snap_number,
       HORIZON_PROTOTYPE_ARGS)
{

  /* We loop over the horizons, look at the ones that are valid,
   * and take the Euclidean distance from a sphere defined by
   * the minimum radius. Then, we take the minimum across all
   * the horizons. If there are no horizons, then we set eps
   * to a very large value. */

  real eps = K(1e4);
  real eps_tmp;

  real4 centroid;
  real radius_max;

  /* One per each horizon */
  compute_eps_hor(1);
  compute_eps_hor(2);
  compute_eps_hor(3);

  return eps;
}


/* real */
/* geteps(real4 q, const whole snap_number, */
/*        HORIZON_PROTOTYPE_ARGS) */
/* { */
/* 	/\* return sqrt(getrr(q)) - (1.0 + sqrt(1.0 - a_spin * a_spin)); *\/ */
/* } */

real4
down(real4 q, real4 u, SPACETIME_PROTOTYPE_ARGS)
{
	real  aa = a_spin * a_spin;
	real  zz = q.s3 * q.s3;
	real  kk = K(0.5) * (q.s1 * q.s1 + q.s2 * q.s2 + zz - aa);
	real  rr = sqrt(kk * kk + aa * zz) + kk;
	real  r  = sqrt(rr);

	real  f  = K(2.0) * rr * r / (rr * rr + aa * zz);
	real  lx = (r * q.s1 + a_spin * q.s2) / (rr + aa);
	real  ly = (r * q.s2 - a_spin * q.s1) / (rr + aa);
	real  lz = q.s3 / r;

	real4 gt = {-1 + f   ,     f*   lx,     f*   ly,     f*   lz};
	real4 gx = {     f*lx, 1 + f*lx*lx,     f*lx*ly,     f*lx*lz};
	real4 gy = {     f*ly,     f*ly*lx, 1 + f*ly*ly,     f*ly*lz};
	real4 gz = {     f*lz,     f*lz*lx,     f*lz*ly, 1 + f*lz*lz};

	return (real4){dot(gt, u),
	               dot(gx, u),
	               dot(gy, u),
	               dot(gz, u)};
}

/**
 ** Sqaure of vector u at the spacetime event q in Kerr-Schild coordiantes
 **
 ** Compute \f$u\cdot u \equiv g_{\alpha\beta} u^\alpha u^\beta\f$,
 ** where \f$g_{\alpha\beta}\f$ is the Cartesian form of the
 ** Kerr-Schild metric.
 **
 ** \return The square of u at q
 **/
real
getuu(struct gr g, SPACETIME_PROTOTYPE_ARGS) /**< state of the ray */
{
	return dot(down(g.q, g.u, SPACETIME_ARGS), g.u);
}

/**
 ** Initial conditions of a ray in an image plane
 **
 ** To perform ray tracing calculations of an image in Kerr spacetime,
 ** we follow Johannsen & Psaltis (2010) and consider an observer
 ** viewing the central black hole from a large distance \p r_obs and
 ** at an inclination angle \p i_obs from its rotation axis (see
 ** Figure 1 of Psaltis & Johannsen 2012).  We set up a virtual image
 ** plane that is perpendicular to the line of sight and centered at
 ** \f$\phi\f$ = \p j_obs of the spacetime.  We define the set of
 ** local Cartesian coordinates (\p alpha, \p beta) on the image plane
 ** such that the \p beta axis is along the same fiducial plane and
 ** the \p alpha axis is perpendicular to it.  These input parameters
 ** define a unique ray, whose initial spacetime position and
 ** wavevector are then computed by icond().
 **
 ** \return The initial conditions of a ray
 **/
struct gr
gr_icond(real r_obs, /**< distance of the image from the black hole */
         real i_obs, /**< inclination angle of the image in degrees */
         real j_obs, /**< azimuthal   angle of the image in degrees */
         real alpha, /**< one of the local Cartesian coordinates */
         real beta,
         SPACETIME_PROTOTYPE_ARGS)  /**< the other  local Cartesian coordinate  */
{
	real  deg2rad = K(3.14159265358979323846264338327950288) / K(180.0);
	real  ci, si  = sincos(deg2rad * i_obs, &ci);
	real  cj, sj  = sincos(deg2rad * j_obs, &cj);

	real  R0 = r_obs * si - beta  * ci; /* cylindrical radius */
	real  z  = r_obs * ci + beta  * si;
	real  y  = R0    * sj + alpha * cj;
	real  x  = R0    * cj - alpha * sj;

	real4 q  = (real4){0, x, y, z};
	real4 u  = (real4){1, si * cj, si * sj, ci};

	real  aa = a_spin * a_spin;
	real  zz = q.s3 * q.s3;
	real  kk = K(0.5) * (q.s1 * q.s1 + q.s2 * q.s2 + zz - aa);
	real  rr = sqrt(kk * kk + aa * zz) + kk;
	real  r  = sqrt(rr);

	real  f  = K(2.0) * rr * r / (rr * rr + aa * zz);
	real  lx = (r * q.s1 + a_spin * q.s2) / (rr + aa);
	real  ly = (r * q.s2 - a_spin * q.s1) / (rr + aa);
	real  lz = q.s3 / r;

	real4 gt = {-1 + f   ,     f*   lx,     f*   ly,     f*   lz};
	real4 gx = {     f*lx, 1 + f*lx*lx,     f*lx*ly,     f*lx*lz};
	real4 gy = {     f*ly,     f*ly*lx, 1 + f*ly*ly,     f*ly*lz};
	real4 gz = {     f*lz,     f*lz*lx,     f*lz*ly, 1 + f*lz*lz};

	real  A  =  gt.s0;
	real  B  =  dot(gt.s123, u.s123) * K(2.0);
	real  C  = (dot(gx.s123, u.s123) * u.s1 +
	            dot(gy.s123, u.s123) * u.s2 +
	            dot(gz.s123, u.s123) * u.s3);

	u.s123 /= -(B + sqrt(B * B - K(4.0) * A * C)) / (K(2.0) * A);

	return (struct gr){q, u};
}

/**
 ** Right hand sides of the geodesic equations in Kerr-Schild coordiantes
 **
 ** One of the breakthroughs we achieve in GRay2 is that, by a series
 ** of mathematical manipulations and regrouping, we significantly
 ** reduce the operation count of the geodesic equations in the
 ** Cartesian Kerr-Schild coordinates.  Let \f$\lambda\f$ be the
 ** affine parameter and \f$\dot{x}^\mu \equiv dx^\mu/d\lambda\f$.  We
 ** show in Chan et al. (2017) that the geodesic equations in the
 ** Cartesian KS coordinates can be optimized to the following form:
 ** \f[
 **  \ddot{x}^\mu = - \left(\eta^{\mu\beta} \dot{x}^\alpha -
 **                         \frac{1}{2}\eta^{\mu\alpha} \dot{x}^\beta\right)
 **                 \dot{x}_{\beta,\alpha} + F l^\mu
 ** \f]
 ** where
 ** \f[
 **   F = f \left(l^\beta \dot{x}^\alpha -
 **               \frac{1}{2}l^\alpha \dot{x}^\beta\right)
 **       \dot{x}_{\beta,\alpha}.
 ** \f]
 ** In this new form, the right hand sides (RHS) of the geodesic
 ** equations have only ~65% more floating-point operations than in
 ** the Boyer-Lindquist coordinates.  Furthermore, the evaluation of
 ** the RHS uses many matrix-vector products, which are optimized in
 ** modern hardwares.
 **
 ** \return The right hand sides of the geodesic equations
 **/
struct gr
gr_rhs(struct gr g, SPACETIME_PROTOTYPE_ARGS) /**< state of the ray */
{
	real4 q = g.q;
	real4 u = g.u;

	real  f,  dx_f,  dy_f,  dz_f;
	real  lx, dx_lx, dy_lx, dz_lx;
	real  ly, dx_ly, dy_ly, dz_ly;
	real  lz, dx_lz, dy_lz, dz_lz;

	real  hDxu, hDyu, hDzu;
	real4 uD;
	real  tmp;

	{
		real dx_r, dy_r, dz_r;
		real r, ir, iss;
		{
			real aa = a_spin * a_spin;
			real rr, tmp2;
			{
				real zz = q.s3 * q.s3;
				real dd;
				{
					real kk = K(0.5) * (q.s1 * q.s1 + q.s2 * q.s2 + zz - aa);
					dd = sqrt(kk * kk + aa * zz);
					rr = dd + kk;
				}
				r  = sqrt(rr);
				ir = K(1.0) / r;
				{
					real ss = rr + aa;
					iss  = K(1.0) / ss;
					tmp  = K(0.5) / (r * dd);
					dz_r = tmp * ss * q.s3;
					tmp *= rr;
				}
				dy_r = tmp * q.s2;
				dx_r = tmp * q.s1;
				tmp  = K(2.0) / (rr + aa * zz / rr);
			}
			tmp2 = K(3.0) - K(2.0) * rr * tmp;
			f    = tmp *  r;
			dx_f = tmp *  dx_r * tmp2;
			dy_f = tmp *  dy_r * tmp2;
			dz_f = tmp * (dz_r * tmp2 - tmp * aa * q.s3 * ir);
		} /* 48 (-8) FLOPs; estimated FLoating-point OPerations, the number
		     in the parentheses is (the negative of) the number of FMA */
		{
			real m2r  = K(-2.0) * r;
			real issr = iss     * r;
			real issa = iss     * a_spin;

			lx    = iss * (q.s1 * r + q.s2 * a_spin);
			tmp   = iss * (q.s1 + m2r * lx);
			dx_lx = tmp * dx_r + issr;
			dy_lx = tmp * dy_r + issa;
			dz_lx = tmp * dz_r;

			ly    = iss * (q.s2 * r - q.s1 * a_spin);
			tmp   = iss * (q.s2 + m2r * ly);
			dx_ly = tmp * dx_r - issa;
			dy_ly = tmp * dy_r + issr;
			dz_ly = tmp * dz_r;

			lz    = q.s3 * ir;
			tmp   = -lz * ir;
			dx_lz = tmp * dx_r;
			dy_lz = tmp * dy_r;
			dz_lz = tmp * dz_r + ir;
		} /* 35 (-9) FLOPs */
	}

	{
		real  flu;
		real4 Dx, Dy, Dz;
		{
			real lu = u.s0 + lx * u.s1 + ly * u.s2 + lz * u.s3;
			flu   = f * lu;
			Dx.s0 = dx_f * lu + f * (dx_lx * u.s1 + dx_ly * u.s2 + dx_lz * u.s3);
			Dy.s0 = dy_f * lu + f * (dy_lx * u.s1 + dy_ly * u.s2 + dy_lz * u.s3);
			Dz.s0 = dz_f * lu + f * (dz_lx * u.s1 + dz_ly * u.s2 + dz_lz * u.s3); /* 31 (-12) FLOPs */
		}
		Dx.s1 = Dx.s0 * lx + flu * dx_lx;
		Dx.s2 = Dx.s0 * ly + flu * dx_ly;
		Dx.s3 = Dx.s0 * lz + flu * dx_lz; /* 9 (-3) FLOPs */

		Dy.s1 = Dy.s0 * lx + flu * dy_lx;
		Dy.s2 = Dy.s0 * ly + flu * dy_ly;
		Dy.s3 = Dy.s0 * lz + flu * dy_lz; /* 9 (-3) FLOPs */

		Dz.s1 = Dz.s0 * lx + flu * dz_lx;
		Dz.s2 = Dz.s0 * ly + flu * dz_ly;
		Dz.s3 = Dz.s0 * lz + flu * dz_lz; /* 9 (-3) FLOPs */

		hDxu = K(0.5) * dot(Dx, u);
		hDyu = K(0.5) * dot(Dy, u);
		hDzu = K(0.5) * dot(Dz, u); /* 24 (-9) FLOPs */

		uD  = u.s1 * Dx + u.s2 * Dy + u.s3 * Dz; /* 20 (-8) FLOPs */

		tmp = f * (-uD.s0 + lx * (uD.s1 - hDxu) + ly * (uD.s2 - hDyu) + lz * (uD.s3 - hDzu)); /* 10 (-3) FLOPs */
	}

	{
		real4 a = {
			       uD.s0 -      tmp,
			hDxu - uD.s1 + lx * tmp,
			hDyu - uD.s2 + ly * tmp,
			hDzu - uD.s3 + lz * tmp
		}; /* 10 (-3) FLOPs */

		#if coordinate_time
		return (struct gr){u, a - a.s0 * u};
		#else
		return (struct gr){u, a};
		#endif
	}
}
