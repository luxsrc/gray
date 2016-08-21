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
#include <lux.h>
#include "gray.h"

int
exec(Lux_job *ego)
{
	struct options *opts = &EGO->options;
	Lux_opencl     *ocl  =  EGO->ocl;

	const size_t gsz[] = {opts->h_rays, opts->w_rays};
	const size_t bsz[] = {1, 1};

	const double dt = 0.1;
	size_t       i;

	lux_debug("GRay2: executing job %p\n", ego);

	/* TODO: check errors */
	clSetKernelArg(EGO->evol, 0, sizeof(cl_mem), &EGO->data);
	clSetKernelArg(EGO->evol, 1, sizeof(double), &dt);

	for(i = 0; i < 10; ++i) {
		lux_print("%zu: %4.1f -> %4.1f", i, i*dt, (i+1)*dt);
		ocl->exec(ocl, EGO->evol, 2, gsz, bsz);
		lux_print(": DONE\n");
	}

	return EXIT_SUCCESS;
}
