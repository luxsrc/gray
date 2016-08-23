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
 ** Classical 4th-order Runge-Kutta integrator
 **
 ** GRay2 uses OpenCL's just-in-time compilation feature to implement
 ** a run-time configurable algorithms.  In this file we implement the
 ** classical 4th-order Runge-Kutta integrator in integrate().
 **/

/**
 ** OpenCL implementation of the classical 4th-order Runge-Kutta integrator
 **
 ** Assuming rhs() is provided, this function performs the classical
 ** 4th-order Runge-Kutta integrator with a single step size dt.
 **
 ** \return The new state
 **/
real8
integrate(real8 s,  /**< State of the ray */
          real  dt) /**< Step size        */
{
	real8 k1 = dt * rhs(s              );
	real8 k2 = dt * rhs(s + K(0.5) * k1);
	real8 k3 = dt * rhs(s + K(0.5) * k2);
	real8 k4 = dt * rhs(s +          k3);
	return s + (k1 + K(2.0) * (k2 + k3) + k4) / K(6.0);
}
