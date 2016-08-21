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

/**
 ** OpenCL implementation of the classical 4th-order Runge-Kutta integrator
 **
 ** Assuming rhs() is provided, this function performs the classical
 ** 4th-order Runge-Kutta integrator with a single step size dt.
 **
 ** \return The new state
 **/
double8
integrate(double8 s, double dt)
{
	double8 k1 = dt * rhs(s           );
	double8 k2 = dt * rhs(s + 0.5 * k1);
	double8 k3 = dt * rhs(s + 0.5 * k2);
	double8 k4 = dt * rhs(s +       k3);
	return s + (k1 + 2.0 * (k2 + k3) + k4) / 6.0;
}
