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

#include "initcond.h"

#include <lux/mangle.h>
#include <lux/switch.h>
#include <lux/zalloc.h>

#include "../infcam_opts.h"

struct infcam {
	Lux_gray_initcond  super;
	size_t             nque;
	cl_command_queue  *que;
};

#define EGO ((struct infcam *)ego)

static int
init(Lux_gray_initcond *ego, cl_mem rays)
{
	/* TODO: initialize cl_mem */
	return 0;
}

void *
LUX_MKMOD(const void *opts)
{
	void *ego;

	lux_debug("GRay2:infcam: constructing an instance with options %p\n", opts);

	ego = zalloc(sizeof(struct infcam));
	if(ego) {
		struct infcam_opts *o = ((Lux_gray_initcond_opts*)opts)->opts;

		EGO->super.init     = init;
		EGO->super.n_width  = o->n_width;
		EGO->super.n_height = o->n_height;

		EGO->nque = ((Lux_gray_initcond_opts*)opts)->nque;
		EGO->que  = ((Lux_gray_initcond_opts*)opts)->que;
	}
	return ego;
}

void
LUX_RMMOD(void *ego)
{
	lux_debug("GRay2:infcam: destructing instance %p\n", ego);

	free(ego);
}
