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
 ** Generic driver kernels
 **
 ** GRay2 uses OpenCL's just-in-time compilation feature to implement
 ** run-time configurable algorithms.  In this file we implement
 ** generic driver kernels icond_drv() and evolve_drv() that uses
 ** IDX(h, k) to access global memory.
 **
 ** We use the index convention `h`, `i`, `j`, `k` for time and the
 ** three spactial coordinates, respectively.  We use `s` to index the
 ** record/field.  These indices may be prefixed by `g` for global
 ** indices, `l` for local indices, etc.
 **/

/** OpenCL driver kernel for initializing states */
__kernel void
icond_drv(__global real *data,  /**< States of the rays                      */
          __global real *info,  /**< Diagnostic information                  */
          const    real  w_img, /**< Width  of the image in \f$GM/c^2\f$     */
          const    real  h_img, /**< Height of the image in \f$GM/c^2\f$     */
          const    real  r_obs, /**< Distance of the image from black hole   */
          const    real  i_obs, /**< Inclination     of the image in degrees */
          const    real  j_obs, /**< Azimuthal angle of the image in degrees */
          __local  real *scratch)
{
	const size_t gj = get_global_id(0); /* for h, slowest changing index */
	const size_t gi = get_global_id(1); /* for w, fastest changing index */
	const size_t g  = gi + gj * w_rays;

	if(gi < w_rays && gj < h_rays) {
		real8 d;
		int   s;

		/* Compute initial conditions from parameters */
		real alpha = ((gi + 0.5) / w_rays - 0.5) * w_img;
		real beta  = ((gj + 0.5) / h_rays - 0.5) * h_img;
		d = icond(r_obs, i_obs, j_obs, alpha, beta);

		/* Output to global array */
		for(s = 0; s < n_data; ++s)
			DATA(g, s) = ((real *)&d)[s];

		for(s = 0; s < n_info; ++s)
			INFO(g, s) = getuu(d.s0123, d.s4567);
	}
}

/** OpenCL driver kernel for integrating the geodesic equations */
__kernel void
evolve_drv(__global real *data,  /**< States of the rays     */
           __global real *info,  /**< Diagnostic information */
           const    real  dt,    /**< Step size              */
           const    whole n_sub, /**< Number of sub-steps    */
           __local  real *scratch,
		   const    real8 bounding_box, /**< Max coordinates of the grid    */
		   __read_only image3d_t Gamma_ttt,
		   __read_only image3d_t Gamma_ttx,
		   __read_only image3d_t Gamma_tty,
		   __read_only image3d_t Gamma_ttz,
		   __read_only image3d_t Gamma_txx,
		   __read_only image3d_t Gamma_txy,
		   __read_only image3d_t Gamma_txz,
		   __read_only image3d_t Gamma_tyy,
		   __read_only image3d_t Gamma_tyz,
		   __read_only image3d_t Gamma_tzz,
		   __read_only image3d_t Gamma_xtt,
		   __read_only image3d_t Gamma_xtx,
		   __read_only image3d_t Gamma_xty,
		   __read_only image3d_t Gamma_xtz,
		   __read_only image3d_t Gamma_xxx,
		   __read_only image3d_t Gamma_xxy,
		   __read_only image3d_t Gamma_xxz,
		   __read_only image3d_t Gamma_xyy,
		   __read_only image3d_t Gamma_xyz,
		   __read_only image3d_t Gamma_xzz,
		   __read_only image3d_t Gamma_ytt,
		   __read_only image3d_t Gamma_ytx,
		   __read_only image3d_t Gamma_yty,
		   __read_only image3d_t Gamma_ytz,
		   __read_only image3d_t Gamma_yxx,
		   __read_only image3d_t Gamma_yxy,
		   __read_only image3d_t Gamma_yxz,
		   __read_only image3d_t Gamma_yyy,
		   __read_only image3d_t Gamma_yyz,
		   __read_only image3d_t Gamma_yzz,
		   __read_only image3d_t Gamma_ztt,
		   __read_only image3d_t Gamma_ztx,
		   __read_only image3d_t Gamma_zty,
		   __read_only image3d_t Gamma_ztz,
		   __read_only image3d_t Gamma_zxx,
		   __read_only image3d_t Gamma_zxy,
		   __read_only image3d_t Gamma_zxz,
		   __read_only image3d_t Gamma_zyy,
		   __read_only image3d_t Gamma_zyz,
		   __read_only image3d_t Gamma_zzz)
{
	const size_t gj = get_global_id(0); /* for h, slowest changing index */
	const size_t gi = get_global_id(1); /* for w, fastest changing index */
	const size_t g  = gi + gj * w_rays;

	if(gi < w_rays && gj < h_rays) {
		real8 d;
		int   s, h;

		/* Input from global array */
		for(s = 0; s < n_data; ++s)
			((real *)&d)[s] = DATA(g, s);

		/* Substepping */
		for(h = 0; h < n_sub; ++h)
			d = integrate(d,
						  dt / n_sub,
						  bounding_box,
						  Gamma_ttt,
						  Gamma_ttx,
						  Gamma_tty,
						  Gamma_ttz,
						  Gamma_txx,
						  Gamma_txy,
						  Gamma_txz,
						  Gamma_tyy,
						  Gamma_tyz,
						  Gamma_tzz,
						  Gamma_xtt,
						  Gamma_xtx,
						  Gamma_xty,
						  Gamma_xtz,
						  Gamma_xxx,
						  Gamma_xxy,
						  Gamma_xxz,
						  Gamma_xyy,
						  Gamma_xyz,
						  Gamma_xzz,
						  Gamma_ytt,
						  Gamma_ytx,
						  Gamma_yty,
						  Gamma_ytz,
						  Gamma_yxx,
						  Gamma_yxy,
						  Gamma_yxz,
						  Gamma_yyy,
						  Gamma_yyz,
						  Gamma_yzz,
						  Gamma_ztt,
						  Gamma_ztx,
						  Gamma_zty,
						  Gamma_ztz,
						  Gamma_zxx,
						  Gamma_zxy,
						  Gamma_zxz,
						  Gamma_zyy,
						  Gamma_zyz,
						  Gamma_zzz);

		/* Output to global array */
		for(s = 0; s < n_data; ++s)
			DATA(g, s) = ((real *)&d)[s];

		for(s = 0; s < n_info; ++s)
			INFO(g, s) = getuu(d.s0123, d.s4567);
	}
}
