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

#include <lux/hdf5.h>
#include <lux/mangle.h>
#include <lux/planner.h>
#include <lux/switch.h>
#include <lux/zalloc.h>

#include <stdio.h>

#include "Kerr_rap.h"

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
	Lux_planner       *gi = NULL;
	Lux_gray_initcond *ic = NULL;

	lux_debug("GRay2: initializing instance %p\n", ego);

	EGO->t  = EGO->gray.t_init;
	EGO->dt = EGO->gray.dt_dump;
	EGO->i  = EGO->gray.i_init;
	EGO->n  = EGO->gray.n_dump;

	lux_print("GRay2:init: setup opencl module\n");
	{
		struct LuxOopencl opts = OPENCL_NULL;
		opts.iplf    = EGO->gray.i_platform;
		opts.idev    = EGO->gray.i_device;
		opts.devtype = EGO->gray.device_type;
		EGO->ocl = lux_load("opencl", &opts);
	}

	EGO->io = lux_load("hdf5", NULL);

	lux_print("GRay2:init: initcond:ic: %s\n", EGO->gray.initcond);
	{
		Lux_gray_initcond_opts opts = {
			EGO->ocl->nque,
			EGO->ocl->que,
			&EGO->initcond
		};

		char buf[256];
		sprintf(buf, "sim/gray/%s", EGO->gray.initcond);

		ic = lux_load(buf, &opts);
		if(!ic)
			return -1;
	}

	lux_print("GRay2:init: allocate memory\n");
	EGO->rays      = dmk(EGO->ocl, real[8], ic->n_width * ic->n_height);
	EGO->rays_host = palloc(real, ic->n_width, ic->n_height, 8);

	lux_print("GRay2:init: initialize rays\n");
	(void)ic->init(ic, EGO->rays.data);

	lux_print("GRay2:init: spacetime:st: %s\n", EGO->gray.spacetime);
	/* TODO: take full advantage of dynamic module and avoid switch */
	SWITCH {
	MATCH(gray.spacetime, "Kerr")
		Lux_Kerr_problem prob = {
			EGO->ocl->nque,
			EGO->ocl->que,
			dgetn(EGO->rays, 0),
			EGO->spacetime.Kerr.a_spin,
			-1.0,
			EGO->rays.data
		};

		char buf[256];
		sprintf(buf, "sim/gray/%s", EGO->gray.spacetime);
		gi = lux_load("planner", buf);

		EGO->gi = gi->plan(gi, (Lux_problem *)&prob, LUX_PLAN_DEFAULT);
	DEFAULT
		lux_fatal("Unknown spacetime configuration \"%s\"\n",
		          EGO->gray.spacetime);
	}

	if(gi)
		lux_unload(gi);
	if(ic)
		lux_unload(ic);

	return 0;
}

static int
exec(Lux_job *ego)
{
	lux_debug("GRay2: executing instance %p\n", ego);

	while(EGO->i < EGO->n) {
		size_t next   = EGO->i + 1;
		double t      = EGO->t;
		double target = EGO->dt * next;

		Lux_file *file;
		char buf[256];

		lux_print("%zu: %4.1f -> %4.1f", next, t, target);

		/* TODO: EGO->gi->exec(EGO->gi); */

		EGO->ocl->d2h(EGO->ocl,
			EGO->rays_host,
			EGO->rays.data, dope_getsz(EGO->rays.dope));

		sprintf(buf, EGO->gray.rayfile, next);
		file = EGO->io(buf, H5F_ACC_EXCL);
		file->write_pa(file, "/rays", typecodeof(real), EGO->rays_host);

		lux_print(": DONE\n");

		EGO->i = next;
		EGO->t = target;
	}

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

	pfree(EGO->rays_host);
	drm(EGO->ocl, EGO->rays);
	lux_unload(EGO->ocl);

	free(ego);
}
