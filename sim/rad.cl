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
 ** Radiative transfer
 **
 ** Radiative transfer related functions such as the emission and
 ** extinction (absorption) coefficients.
 **/

struct rad {
	real I  [n_freq];
	real tau[n_freq];
};

/**
 ** Initial conditions of radiation properties
 **
 ** \return The initial conditions of radiation properties
 **/
struct rad
rad_icond(void)
{
	return (struct rad){{0}};
}

/**
 ** Right hand sides of the radiative transfer equation
 **
 ** \return The right hand sides of the radiative transfer equation
 **/
struct rad
rad_rhs(struct ray s,
        struct rad t)
{
	struct rad out = {{0}};

	for(whole i; i < n_freq; ++i) {
		/* Radiative transfer */
	}

	return out;
}
