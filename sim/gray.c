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
#include <lux/check.h>
#include <lux/mangle.h>
#include <lux/zalloc.h>
#include "gray.h"

#define EGO ((struct gray *)ego)
#define CKR lux_check_func_success

static int
conf(Lux_job *ego, const char *restrict arg)
{
	lux_debug("GRay2: configuring job %p with argument \"%s\"\n", ego, arg);

	return options_config(&EGO->options, arg);
}

static int
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

static int
exec(Lux_job *ego)
{
	lux_debug("GRay2: executing job %p\n", ego);

	return EXIT_SUCCESS;
}

void *
LUX_MKMOD(const void *opts)
{
	void *ego;

	lux_debug("GRay2: constructing with configuration %p\n", opts);

	ego = zalloc(sizeof(struct gray));
	if(ego) {
		EGO->super.conf = conf;
		EGO->super.init = init;
		EGO->super.exec = exec;
		options_init(&EGO->options);
	}
	return ego;
}

void
LUX_RMMOD(void *ego)
{
	Lux_opencl *ocl = EGO->ocl;

	lux_debug("GRay2: destructing instance %p\n", ego);

	if(EGO->data)
		ocl->rm(EGO->data);
	if(EGO->ocl)
		lux_unload(EGO->ocl);
	free(ego);
}
