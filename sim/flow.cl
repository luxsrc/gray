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
 ** Flow models
 **
 ** Flow models of the accretion flows, which may be interpolation of
 ** GRMHD simulations or analytical models.
 **/

struct flow {
	real ne;
	real te;
	real b;
	real bkcos;
	real shift;
};

struct flow
getflow(real4 q, real4 k, SPACETIME_PROTOTYPE_ARGS)
{
	struct flow f;

	real4 u = {1, 0, 0, 0};
	real4 b; /* \todo check b is defined consistently;
				to compute synchrotron radiation, all we
				care are the magnitude of the magnetic
				field and the angle cosine between the
				magnetic field and the photon momentum
				vector. */

	b.s0 = interpolate(q, bounding_box, num_points, bt_t1, bt_t2);
	b.s1 = interpolate(q, bounding_box, num_points, bx_t1, bx_t2);
	b.s2 = interpolate(q, bounding_box, num_points, by_t1, by_t2);
	b.s3 = interpolate(q, bounding_box, num_points, bz_t1, bz_t2);

	f.ne = interpolate(q, bounding_box, num_points, rho_t1, rho_t2);
	f.te = interpolate(q, bounding_box, num_points, temperature_t1, temperature_t2);
	f.b  = sqrt(dot(b, down(q, b, SPACETIME_ARGS)));

	f.shift = -dot(k, u);
	f.bkcos =  dot(k, b) / (f.shift * f.b + (real)EPSILON);

	return f;
}
