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
init(Lux_job *ego)
{
	struct options *opts = &EGO->options;

	size_t ray_sz = sizeof(real) * 8;
	size_t n_rays = opts->w_rays * opts->h_rays;

	Lux_opencl *ocl;
	cl_mem      data;

	lux_debug("GRay2: initializing job %p\n", ego);

	CKR(ocl  = lux_load("opencl", NULL),                                cleanup1);
	CKR(data = ocl->mk(ocl->super, CL_MEM_READ_WRITE, ray_sz * n_rays), cleanup2);

	EGO->ocl  = ocl;
	EGO->data = data;
	return EXIT_SUCCESS;

 cleanup2:
	lux_unload(ocl);
 cleanup1:
	return EXIT_FAILURE;
}
