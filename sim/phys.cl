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
 ** Composing different physics modules
 **
 ** This file contains functions that compose different physics, e.g.,
 ** geodesic integration, radiative transfer, together to form the
 ** initial conditions and the right hand side of the full
 ** differential equation.
 **/

struct state {
	struct ray r;
#if n_freq > 0
	real I  [n_freq];
	real tau[n_freq];
#endif
};

struct state
icond(real r_obs, /**< Distance of the observer from the black hole */
      real i_obs, /**< Inclination angle of the observer in degrees */
      real j_obs, /**< Azimuthal   angle of the observer in degrees */
      real alpha, /**< One of the local Cartesian coordinates       */
      real beta)  /**< The other  local Cartesian coordinate        */
{
	return (struct state){
		ray_icond(r_obs, i_obs, j_obs, alpha, beta)
#if n_freq > 0
		/** \todo initialize radiative transfer */
#endif
	};
}

struct state
rhs(struct state s) /**< State of the ray */
{
#if n_freq > 0
	for(whole i; i < n_freq; ++i) {
		/* Radiative transfer */
	}
#endif

	return (struct state){
		ray_rhs(s.r)
#if n_freq > 0
		/** \todo initialize radiative transfer */
#endif
	};
}
