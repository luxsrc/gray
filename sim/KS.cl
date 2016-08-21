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
double8
icond(double r_obs, double i_obs, double j_obs, double alpha, double beta)
{
	double  ci, si = sincos(M_PI * i_obs / 180.0, &ci);
	double  cj, sj = sincos(M_PI * j_obs / 180.0, &cj);

	double  R = r_obs * si - beta  * ci; /* cylindrical radius */
	double  z = r_obs * ci + beta  * si;
	double  y = R     * sj - alpha * cj;
	double  x = R     * cj + alpha * sj;

	double4 q = (double4){0.0, x, y, z};
	double4 u = (double4){1.0, si * cj, si * sj, ci};

	return (double8){q, u};
}

static double8
rhs(double8 s)
{
	/* TODO: actually implement the right hand side */
	return (double8){s.s4567, 0, 0, 0, 0};
}
