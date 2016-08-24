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

#include <lux/mangle.h>
#include <lux/zalloc.h>

#include <stdio.h>

static int
_conf(Lux_job *ego, const char *restrict arg)
{
	/** \page newopts New Run-Time Options
	 **
	 ** Turn hard-wired constants into run-time options
	 **
	 ** GRay2 uses the lux framework and hence follows lux's
	 ** approach to support many run-time options.  To turn
	 ** hard-wired constants into run-time options, one needs to
	 **
	 ** -# Add an option table ".opts" file in the "sim/"
	 **    directory.
	 ** -# Embed the automatically generated structure to `struct
	 **    gray` in "sim/gray.h"
	 ** -# Logically `&&` the automatically generated configure
	 **    function to the return values of `_conf()` in
	 **    "sim/gray.c".
	 **/
	lux_debug("GRay2: configuring instance %p with \"%s\"\n", ego, arg);

	return icond_config(&EGO->icond, arg) &&
	       param_config(&EGO->param, arg) &&
	       setup_config(&EGO->setup, arg);
}

static int
_init(Lux_job *ego)
{
	Lux_opencl *ocl; /* to be loaded */

	struct param *p = &EGO->param;
	struct setup *s = &EGO->setup;

	const size_t sz     = s->precision;
	const size_t n_rays = p->h_rays * p->w_rays;
	const size_t n_vars = p->n_freq * 2 + 8;

	cl_mem_flags flags  = CL_MEM_READ_WRITE;

	lux_debug("GRay2: initializing instance %p\n", ego);

	CKR(EGO->ocl    = ocl    = build(ego),                       cleanup1);
	CKR(EGO->data   = ocl->mk(ocl, flags, sz * n_rays * n_vars), cleanup2);
	CKR(EGO->info   = ocl->mk(ocl, flags, sz * n_rays),          cleanup3);
	CKR(EGO->evolve = ocl->mkkern(ocl, "evolve_drv"),            cleanup4);

	return EXIT_SUCCESS;

 cleanup4:
	ocl->rm(ocl, EGO->info);
 cleanup3:
	ocl->rm(ocl, EGO->data);
 cleanup2:
	lux_unload(EGO->ocl);
 cleanup1:
	return EXIT_FAILURE;
}

static int
_exec(Lux_job *ego)
{
	struct param *p = &EGO->param;

	const  size_t n_rays = p->h_rays * p->w_rays;
	const  double dt     = -1.0;
	const  size_t n_sub  = 1024;

	size_t i;

	lux_debug("GRay2: executing instance %p\n", ego);

	icond(ego);
	dump(ego, 0);

	for(i = 0; i < 10; ++i) {
		double ns;
		lux_print("%zu: %4.1f -> %4.1f", i, i*dt, (i+1)*dt);
		ns = evolve(ego);
		dump(ego, i+1);
		lux_print(": DONE (%.3gns/step/ray)\n", ns/n_sub/n_rays);
	}

	return EXIT_SUCCESS;
}

void *
LUX_MKMOD(const void *opts)
{
	void *ego;

	lux_debug("GRay2: constructing with options %p\n", opts);

	ego = zalloc(sizeof(struct gray));
	if(ego) {
		EGO->super.conf = _conf;
		EGO->super.init = _init;
		EGO->super.exec = _exec;
		icond_init(&EGO->icond);
		param_init(&EGO->param);
		setup_init(&EGO->setup);
	}
	return ego;
}

void
LUX_RMMOD(void *ego)
{
	Lux_opencl *ocl = EGO->ocl;

	lux_debug("GRay2: destructing instance %p\n", ego);

	if(EGO->evolve)
		ocl->rmkern(ocl, EGO->evolve);
	if(EGO->info)
		ocl->rm(ocl, EGO->info);
	if(EGO->data)
		ocl->rm(ocl, EGO->data);
	if(EGO->ocl)
		lux_unload(EGO->ocl);
	free(ego);
}
