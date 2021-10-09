/*
 * Copyright (C) 2021 Gabriele Bozzola and Chi-kwan Chan
 * Copyright (C) 2021 Steward Observatory
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

#include <lux/mangle.h>
#include <lux/switch.h>
#include <lux/zalloc.h>

#define EGO ((struct gray *)ego)

#define MATCH(opt, str) CASE(!strcmp(EGO->opt, str))

static int
conf(Lux_job *ego, const char *restrict arg)
{
	const char *spacetime_org = EGO->gray.spacetime;
	const char *initcond_org  = EGO->gray.initcond;
	int status;

	lux_debug("GRay2: configuring instance %p with \"%s\"\n", ego, arg);

	status = gray_config(&EGO->gray, arg);

	/* TODO: take full advantage of dynamic module and avoid switch */
	SWITCH {
	MATCH(gray.spacetime, "Kerr")
		if(EGO->gray.spacetime != spacetime_org)
			Kerr_init(&EGO->spacetime.Kerr);
		else if(status)
			status = Kerr_config(&EGO->spacetime.Kerr, arg);
	DEFAULT
		lux_fatal("Unknown spacetime configuration \"%s\"\n",
		          EGO->gray.spacetime);
	}

	/* TODO: take full advantage of dynamic module and avoid switch */
	SWITCH {
	MATCH(gray.initcond, "infcam")
		if(EGO->gray.initcond != initcond_org)
			infcam_init(&EGO->initcond.infcam);
		else if(status)
			status = infcam_config(&EGO->initcond.infcam, arg);
	DEFAULT
		lux_fatal("Unknown initial conditions for rays \"%s\"\n",
		          EGO->gray.initcond);
	}

	return status;
}

static int
init(Lux_job *ego)
{
	lux_debug("GRay2: initializing instance %p\n", ego);

	lux_print("Setup OpenCL\n");
	{
		struct LuxOopencl opts = OPENCL_NULL;
		opts.iplf    = EGO->gray.i_platform;
		opts.idev    = EGO->gray.i_device;
		opts.devtype = EGO->gray.device_type;
		EGO->ocl = lux_load("opencl", &opts);
	}

	lux_print("spacetime:st: %s\n", EGO->gray.spacetime);
	lux_print("initcond:ic: %s\n",  EGO->gray.initcond);

	/* TODO: take full advantage of dynamic module and avoid switch */
	SWITCH {
	MATCH(gray.initcond, "infcam")
		EGO->n_rays = (
			EGO->initcond.infcam.n_width  *
			EGO->initcond.infcam.n_height);
	}

	EGO->rays = EGO->ocl->mk(
		EGO->ocl,
		8 * sizeof(real) * EGO->n_rays,
		CL_MEM_READ_WRITE);

	return 0;
}

static int
exec(Lux_job *ego)
{
	lux_debug("GRay2: executing instance %p\n", ego);

	return 0;
}

void *
LUX_MKMOD(const void *opts)
{
	void *ego;

	lux_debug("GRay2: constructing an instance with options %p\n", opts);

	ego = zalloc(sizeof(struct gray));
	if(ego) {
		EGO->super.conf = conf;
		EGO->super.init = init;
		EGO->super.exec = exec;

		gray_init(&EGO->gray);
		infcam_init(&EGO->initcond.infcam);
	}
	return ego;
}

void
LUX_RMMOD(void *ego)
{
	lux_debug("GRay2: destructing instance %p\n", ego);

	EGO->ocl->rm(EGO->ocl, EGO->rays);
	lux_unload(EGO->ocl);

	free(ego);
}
