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
__kernel void
init(__global double  *diagno,
     __global double8 *states,
     const double w_img, const double h_img,
     const double r_obs, const double i_obs, const double j_obs)
{
	const size_t j = get_global_id(0); /* for h, slowest changing index */
	const size_t i = get_global_id(1); /* for w, fastest changing index */
	const size_t h = i + j * w_rays;

	if(i < w_rays && j < h_rays) {
		/* Compute initial conditions from parameters */
		double  alpha = ((i + 0.5) / w_rays - 0.5) * w_img;
		double  beta  = ((j + 0.5) / h_rays - 0.5) * h_img;
		double8 s     = icond(r_obs, i_obs, j_obs, alpha, beta);

		/* Output to global array */
		diagno[h] = getuu(s.s0123, s.s4567);
		states[h] = s;
	}
}

__kernel void
evol(__global double  *diagno,
     __global double8 *states,
     const double dt, const size_t n_sub)
{
	const size_t j = get_global_id(0); /* for h, slowest changing index */
	const size_t i = get_global_id(1); /* for w, fastest changing index */
	const size_t h = i + j * w_rays;

	if(i < w_rays && j < h_rays) {
		/* Input from global array */
		double8 s = states[h];

		/* Substepping */
		double ddt = dt / n_sub;
		size_t i;
		for(i = 0; i < n_sub; ++i)
			s = integrate(s, ddt);

		/* Output to global array */
		diagno[h] = getuu(s.s0123, s.s4567);
		states[h] = s;
	}
}
