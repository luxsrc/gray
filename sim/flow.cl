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
getflow(real4 q)
{
	struct flow f;

	real  aa = a_spin * a_spin;
	real  zz = q.s3 * q.s3;
	real  kk = K(0.5) * (q.s1 * q.s1 + q.s2 * q.s2 + zz - aa);
	real  dd = sqrt(kk * kk + aa * zz);
	real  rr = dd + kk;
	real  r  = sqrt(rr);

	if(r > 2.0)
		f.ne = 1.0e9 / r;
	else
		f.ne = 0.0;

	f.te = 1e12;
	f.b  = 1e3;

	f.bkcos = 1.0;
	f.shift = 1.0;

	return f;
}
