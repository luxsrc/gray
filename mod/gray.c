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
#include <lux/zalloc.h>

#define EGO ((struct gray *)ego)

static int
conf(Lux_job *ego, const char *restrict arg)
{
	lux_debug("GRay2: configuring instance %p with \"%s\"\n", ego, arg);

	return gray_config(&EGO->opts, arg);
}

static int
init(Lux_job *ego)
{
	lux_debug("GRay2: initializing instance %p\n", ego);

	lux_print("Setup OpenCL\n");
	{
		struct LuxOopencl opts = OPENCL_NULL;
		opts.iplf    = EGO->opts.i_platform;
		opts.idev    = EGO->opts.i_device;
		opts.devtype = EGO->opts.device_type;
		EGO->ocl = lux_load("opencl", &opts);
	}

	lux_print("spacetime: %s\n", EGO->opts.spacetime);

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

		gray_init(&EGO->opts);
	}
	return ego;
}

void
LUX_RMMOD(void *ego)
{
	lux_debug("GRay2: destructing instance %p\n", ego);

	lux_unload(EGO->ocl);

	free(ego);
}
