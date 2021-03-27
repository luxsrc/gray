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
	struct gr g;
	struct rt r;
};

struct state
icond(real r_obs, /**< distance of the image from the black hole */
      real i_obs, /**< inclination angle of the image in degrees */
      real j_obs, /**< azimuthal   angle of the image in degrees */
      real alpha, /**< one of the local Cartesian coordinates */
      real beta)  /**< the other  local Cartesian coordinate  */
{
	return (struct state){
		gr_icond(r_obs, i_obs, j_obs, alpha, beta),
		rt_icond()
	};
}

real
getdt(struct gr g, real dt)
{
	real r   = sqrt(getrr(g.q));
	real eps = geteps(g.q);

	if(eps < 0.01) /* stop near horizon */
		return 0;

	if(dot(g.q.s123, g.q.s123) < 0 && r > K(1e3)) /* stop outside domain */
		return 0;

	if(fabs(dt) > 0.01 * eps * eps)
		return sign(dt) * 0.01 * eps * eps;
	else
		return dt;
}

struct state
rhs(struct state s, SPACETIME_PROTOTYPE_ARGS) /**< state of the ray */
{
	real4 q = s.g.q;
	real4 k = down(q, s.g.u);
	return (struct state){gr_rhs(s.g, SPACETIME_ARGS), rt_rhs(getflow(q, k))};
}
