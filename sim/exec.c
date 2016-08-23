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
#include "gray.h"
#include <stdio.h>

int
exec(Lux_job *ego)
{
	struct param *p   = &EGO->param;
	Lux_opencl   *ocl =  EGO->ocl;

	const size_t n_rays =  p->h_rays * p->w_rays;
	const size_t gsz[]  = {p->h_rays,  p->w_rays};
	const size_t bsz[]  = {1, 1};

	double dt    = -1.0;
	size_t n_sub = 1024;
	size_t i;

	lux_debug("GRay2: executing job %p\n", ego);

	dump(ego, "0000.raw");

	/** \todo check errors */
	ocl->setM(ocl, EGO->evol, 0, EGO->diag);
	ocl->setM(ocl, EGO->evol, 1, EGO->data);
	ocl->setR(ocl, EGO->evol, 2, dt);
	ocl->setW(ocl, EGO->evol, 3, n_sub);

	for(i = 0; i < 10; ++i) {
		char   buf[64];
		double ns;

		lux_print("%zu: %4.1f -> %4.1f", i, i*dt, (i+1)*dt);

		ns = ocl->exec(ocl, EGO->evol, 2, gsz, bsz);

		snprintf(buf, sizeof(buf), "%04zu.raw", i+1);
		dump(ego, buf);

		lux_print(": DONE (%.3gms/dump/img, "
		                  "%.3gus/step/img, "
		                  "%.3gns/step/ray)\n",
		          1e-6 * ns, 1e-3 * ns / n_sub, ns / n_sub / n_rays);
	}

	return EXIT_SUCCESS;
}
