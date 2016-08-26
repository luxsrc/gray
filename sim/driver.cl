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
icond_drv(__global real *data,  /**< states of the rays                      */
          __global real *info,  /**< diagnostic information                  */
          const    real  w_img, /**< width  of the image in \f$GM/c^2\f$     */
          const    real  h_img, /**< height of the image in \f$GM/c^2\f$     */
          const    real  r_obs, /**< distance of the image from black hole   */
          const    real  i_obs, /**< inclination     of the image in degrees */
          const    real  j_obs, /**< azimuthal angle of the image in degrees */
          __local  real *scratch)
{
	const size_t gj = get_global_id(0); /* for h, slowest changing index */
	const size_t gi = get_global_id(1); /* for w, fastest changing index */
	const size_t g  = gi + gj * w_rays;

	if(gi < w_rays && gj < h_rays) {
		struct state d;
		int s;

		/* Compute initial conditions from parameters */
		real alpha = ((gi + 0.5) / w_rays - 0.5) * w_img;
		real beta  = ((gj + 0.5) / h_rays - 0.5) * h_img;
		d = icond(r_obs, i_obs, j_obs, alpha, beta);

		/* Output to global array */
		for(s = 0; s < n_data; ++s)
			DATA(g, s) = ((real *)&d)[s];

		for(s = 0; s < n_info; ++s)
			INFO(g, s) = getuu(d.r);
	}
}

/** OpenCL driver kernel for integrating the geodesic equations */
__kernel void
evolve_drv(__global real *data,  /**< states of the rays     */
           __global real *info,  /**< diagnostic information */
           const    real  dt,    /**< step size              */
           const    whole n_sub, /**< number of sub-steps    */
           __local  real *scratch)
{
	const size_t gj = get_global_id(0); /* for h, slowest changing index */
	const size_t gi = get_global_id(1); /* for w, fastest changing index */
	const size_t g  = gi + gj * w_rays;

	if(gi < w_rays && gj < h_rays) {
		struct state d;
		int s, h;

		/* Input from global array */
		for(s = 0; s < n_data; ++s)
			((real *)&d)[s] = DATA(g, s);

		/* Substepping */
		for(h = 0; h < n_sub; ++h)
			d = integrate(d, dt / n_sub);

		/* Output to global array */
		for(s = 0; s < n_data; ++s)
			DATA(g, s) = ((real *)&d)[s];

		for(s = 0; s < n_info; ++s)
			INFO(g, s) = getuu(d.r);
	}
}
