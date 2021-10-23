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
 ** A template for flow models
 **
 ** A template for an accretion flows model, which may be
 ** interpolation of GRMHD simulations or analytical models.
 **/

struct flow {
	real ne;
	real te;
	real b;
	real bkcos;
	real shift;
};

struct flow
getflow(real4 q, /* "up"   position 4-vector q^mu */
        real4 k, /* "down" momentum 4-vector k_mu */
	SPACETIME_PROTOTYPE_ARGS)
{
	struct flow f;

	real4 u = {1, 0, 0, 0};
	real4 b = {0, 0, 0, 0}; /* magnetic field four-vector defined as
	                           b^mu = (1/2)
                                           epsilon^{mu,nu,kappa,lambda}
                                           u_nu F_{lambda,kappa},
                                   see Gammie et al. (2003) */

	f.ne = interpolate(q, bounding_box, num_points, rho_t1, rho_t2);
	f.te = 1e12;
	f.b  = 0;

	f.shift = -dot(k, u);
	f.bkcos =  dot(k, b) / (f.shift * f.b + (real)EPSILON);

	return f;
}
