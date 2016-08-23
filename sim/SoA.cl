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
 ** Structure-of-Arrays driver kernels
 **
 ** GRay2 uses OpenCL's just-in-time compilation feature to implement
 ** run-time configurable algorithms.  In this file we implement
 ** Structure-of-Arrays driver kernels icond_drv() and integrate_drv().
 **/

/** OpenCL driver kernel for initializing states */
__kernel void
icond_drv(__global real *diagno, /**< Buffer for storing diagnostic information */
          __global real *states, /**< Buffer for storing the states of the rays */
          const    real  w_img,  /**< Width  of the image in \f$GM/c^2\f$       */
          const    real  h_img,  /**< Height of the image in \f$GM/c^2\f$       */
          const    real  r_obs,  /**< Distance of the image from the black hole */
          const    real  i_obs,  /**< Inclination angle of the image in degrees */
          const    real  j_obs)  /**< Azimuthal   angle of the image in degrees */
{
	const size_t j = get_global_id(0); /* for h, slowest changing index */
	const size_t i = get_global_id(1); /* for w, fastest changing index */
	const size_t h = i + j * w_rays;

	if(i < w_rays && j < h_rays) {
		real8 s;
		int   k;

		/* Compute initial conditions from parameters */
		real alpha = ((i + 0.5) / w_rays - 0.5) * w_img;
		real beta  = ((j + 0.5) / h_rays - 0.5) * h_img;
		s = icond(r_obs, i_obs, j_obs, alpha, beta);

		/* Output to global array */
		diagno[h] = getuu(s.s0123, s.s4567);
		for(k = 0; k < 8; ++k)
			states[k * n_rays + h] = ((real *)&s)[k];
	}
}

/** OpenCL driver kernel for integrating the geodesic equations */
__kernel void
integrate_drv(__global real *diagno, /**< Buffer for storing diagnostic information */
              __global real *states, /**< Buffer for storing the states of the rays */
              const    real  dt,     /**< Step size                                 */
              const    whole n_sub)  /**< Number of sub-steps                       */
{
	const size_t j = get_global_id(0); /* for h, slowest changing index */
	const size_t i = get_global_id(1); /* for w, fastest changing index */
	const size_t h = i + j * w_rays;

	if(i < w_rays && j < h_rays) {
		real8 s;
		int   k;

		/* Input from global array */
		for(k = 0; k < 8; ++k)
			((real *)&s)[k] = states[k * n_rays + h];

		/* Substepping */
		real  ddt = dt / n_sub;
		whole i;
		for(i = 0; i < n_sub; ++i)
			s = integrate(s, ddt);

		/* Output to global array */
		diagno[h] = getuu(s.s0123, s.s4567);
		for(k = 0; k < 8; ++k)
			states[k * n_rays + h] = ((real *)&s)[k];
	}
}
