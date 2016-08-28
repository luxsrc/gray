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
struct state
integrate(struct state s,  /**< state of the ray */
          real         dt) /**< step size        */
{
	real4 q = s.g.q;
	real4 u = s.g.u;

	real  aa = a_spin * a_spin;
	real  zz = q.s3 * q.s3;
	real  kk = K(0.5) * (q.s1 * q.s1 + q.s2 * q.s2 + zz - aa);
	real  dd = sqrt(kk * kk + aa * zz);
	real  rr = dd + kk;
	real  r  = sqrt(rr);

	if(r < 1.0 + sqrt(1.0 - aa)) /* stop inside horizon */
		return s;

	if(q.s1 * u.s1 + q.s2 * u.s2 + q.s3 * u.s3 < 0 && r > K(1e3)) /* outside domain */
		return s;

	if(n_freq) {
		bool done = 1;
		for(whole i = 0; i < n_freq; ++i)
			if(s.r.tau[i] < K(6.90775527898))
				done = 0;
		if(done) /* stop if optically thick */
			return s;
	}

	struct state k1 = rhs(X(E(s)                      ));
	struct state k2 = rhs(X(E(s) + K(0.5) * dt * E(k1)));
	struct state k3 = rhs(X(E(s) + K(0.5) * dt * E(k2)));
	struct state k4 = rhs(X(E(s) +          dt * E(k3)));
	return X(E(s) + dt * (E(k1) + K(2.0) * (E(k2) + E(k3)) + E(k4)) / K(6.0));
}
