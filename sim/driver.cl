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
		   const    int4 num_points,  /**< Number of points on the grid    */
		   __read_only image3d_t Gamma_ttt_t1,
		   __read_only image3d_t Gamma_ttx_t1,
		   __read_only image3d_t Gamma_tty_t1,
		   __read_only image3d_t Gamma_ttz_t1,
		   __read_only image3d_t Gamma_txx_t1,
		   __read_only image3d_t Gamma_txy_t1,
		   __read_only image3d_t Gamma_txz_t1,
		   __read_only image3d_t Gamma_tyy_t1,
		   __read_only image3d_t Gamma_tyz_t1,
		   __read_only image3d_t Gamma_tzz_t1,
		   __read_only image3d_t Gamma_xtt_t1,
		   __read_only image3d_t Gamma_xtx_t1,
		   __read_only image3d_t Gamma_xty_t1,
		   __read_only image3d_t Gamma_xtz_t1,
		   __read_only image3d_t Gamma_xxx_t1,
		   __read_only image3d_t Gamma_xxy_t1,
		   __read_only image3d_t Gamma_xxz_t1,
		   __read_only image3d_t Gamma_xyy_t1,
		   __read_only image3d_t Gamma_xyz_t1,
		   __read_only image3d_t Gamma_xzz_t1,
		   __read_only image3d_t Gamma_ytt_t1,
		   __read_only image3d_t Gamma_ytx_t1,
		   __read_only image3d_t Gamma_yty_t1,
		   __read_only image3d_t Gamma_ytz_t1,
		   __read_only image3d_t Gamma_yxx_t1,
		   __read_only image3d_t Gamma_yxy_t1,
		   __read_only image3d_t Gamma_yxz_t1,
		   __read_only image3d_t Gamma_yyy_t1,
		   __read_only image3d_t Gamma_yyz_t1,
		   __read_only image3d_t Gamma_yzz_t1,
		   __read_only image3d_t Gamma_ztt_t1,
		   __read_only image3d_t Gamma_ztx_t1,
		   __read_only image3d_t Gamma_zty_t1,
		   __read_only image3d_t Gamma_ztz_t1,
		   __read_only image3d_t Gamma_zxx_t1,
		   __read_only image3d_t Gamma_zxy_t1,
		   __read_only image3d_t Gamma_zxz_t1,
		   __read_only image3d_t Gamma_zyy_t1,
		   __read_only image3d_t Gamma_zyz_t1,
		   __read_only image3d_t Gamma_zzz_t1, /* 2 */
		   __read_only image3d_t Gamma_ttt_t2,
		   __read_only image3d_t Gamma_ttx_t2,
		   __read_only image3d_t Gamma_tty_t2,
		   __read_only image3d_t Gamma_ttz_t2,
		   __read_only image3d_t Gamma_txx_t2,
		   __read_only image3d_t Gamma_txy_t2,
		   __read_only image3d_t Gamma_txz_t2,
		   __read_only image3d_t Gamma_tyy_t2,
		   __read_only image3d_t Gamma_tyz_t2,
		   __read_only image3d_t Gamma_tzz_t2,
		   __read_only image3d_t Gamma_xtt_t2,
		   __read_only image3d_t Gamma_xtx_t2,
		   __read_only image3d_t Gamma_xty_t2,
		   __read_only image3d_t Gamma_xtz_t2,
		   __read_only image3d_t Gamma_xxx_t2,
		   __read_only image3d_t Gamma_xxy_t2,
		   __read_only image3d_t Gamma_xxz_t2,
		   __read_only image3d_t Gamma_xyy_t2,
		   __read_only image3d_t Gamma_xyz_t2,
		   __read_only image3d_t Gamma_xzz_t2,
		   __read_only image3d_t Gamma_ytt_t2,
		   __read_only image3d_t Gamma_ytx_t2,
		   __read_only image3d_t Gamma_yty_t2,
		   __read_only image3d_t Gamma_ytz_t2,
		   __read_only image3d_t Gamma_yxx_t2,
		   __read_only image3d_t Gamma_yxy_t2,
		   __read_only image3d_t Gamma_yxz_t2,
		   __read_only image3d_t Gamma_yyy_t2,
		   __read_only image3d_t Gamma_yyz_t2,
		   __read_only image3d_t Gamma_yzz_t2,
		   __read_only image3d_t Gamma_ztt_t2,
		   __read_only image3d_t Gamma_ztx_t2,
		   __read_only image3d_t Gamma_zty_t2,
		   __read_only image3d_t Gamma_ztz_t2,
		   __read_only image3d_t Gamma_zxx_t2,
		   __read_only image3d_t Gamma_zxy_t2,
		   __read_only image3d_t Gamma_zxz_t2,
		   __read_only image3d_t Gamma_zyy_t2,
		   __read_only image3d_t Gamma_zyz_t2,
		   __read_only image3d_t Gamma_zzz_t2)
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
						  num_points,
						  Gamma_ttt_t1,
						  Gamma_ttx_t1,
						  Gamma_tty_t1,
						  Gamma_ttz_t1,
						  Gamma_txx_t1,
						  Gamma_txy_t1,
						  Gamma_txz_t1,
						  Gamma_tyy_t1,
						  Gamma_tyz_t1,
						  Gamma_tzz_t1,
						  Gamma_xtt_t1,
						  Gamma_xtx_t1,
						  Gamma_xty_t1,
						  Gamma_xtz_t1,
						  Gamma_xxx_t1,
						  Gamma_xxy_t1,
						  Gamma_xxz_t1,
						  Gamma_xyy_t1,
						  Gamma_xyz_t1,
						  Gamma_xzz_t1,
						  Gamma_ytt_t1,
						  Gamma_ytx_t1,
						  Gamma_yty_t1,
						  Gamma_ytz_t1,
						  Gamma_yxx_t1,
						  Gamma_yxy_t1,
						  Gamma_yxz_t1,
						  Gamma_yyy_t1,
						  Gamma_yyz_t1,
						  Gamma_yzz_t1,
						  Gamma_ztt_t1,
						  Gamma_ztx_t1,
						  Gamma_zty_t1,
						  Gamma_ztz_t1,
						  Gamma_zxx_t1,
						  Gamma_zxy_t1,
						  Gamma_zxz_t1,
						  Gamma_zyy_t1,
						  Gamma_zyz_t1,
						  Gamma_zzz_t1, /* 2 */
						  Gamma_ttt_t2,
						  Gamma_ttx_t2,
						  Gamma_tty_t2,
						  Gamma_ttz_t2,
						  Gamma_txx_t2,
						  Gamma_txy_t2,
						  Gamma_txz_t2,
						  Gamma_tyy_t2,
						  Gamma_tyz_t2,
						  Gamma_tzz_t2,
						  Gamma_xtt_t2,
						  Gamma_xtx_t2,
						  Gamma_xty_t2,
						  Gamma_xtz_t2,
						  Gamma_xxx_t2,
						  Gamma_xxy_t2,
						  Gamma_xxz_t2,
						  Gamma_xyy_t2,
						  Gamma_xyz_t2,
						  Gamma_xzz_t2,
						  Gamma_ytt_t2,
						  Gamma_ytx_t2,
						  Gamma_yty_t2,
						  Gamma_ytz_t2,
						  Gamma_yxx_t2,
						  Gamma_yxy_t2,
						  Gamma_yxz_t2,
						  Gamma_yyy_t2,
						  Gamma_yyz_t2,
						  Gamma_yzz_t2,
						  Gamma_ztt_t2,
						  Gamma_ztx_t2,
						  Gamma_zty_t2,
						  Gamma_ztz_t2,
						  Gamma_zxx_t2,
						  Gamma_zxy_t2,
						  Gamma_zxz_t2,
						  Gamma_zyy_t2,
						  Gamma_zyz_t2,
						  Gamma_zzz_t2);

		/* Output to global array */
		for(s = 0; s < n_data; ++s)
			DATA(g, s) = ((real *)&d)[s];

		for(s = 0; s < n_info; ++s)
			INFO(g, s) = getuu(d.s0123, d.s4567);
	}
}
