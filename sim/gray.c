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
#include <lux/mangle.h>
#include <lux/zalloc.h>
#include "gray.h"

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

	if(EGO->evol)
		ocl->rmkern(EGO->evol);
	if(EGO->data)
		ocl->rm(EGO->data);
	if(EGO->ocl)
		lux_unload(EGO->ocl);
	free(ego);
}
