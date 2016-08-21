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
#include <stdio.h>

/* TODO: implment load() */

void
dump(Lux_job *ego, const char *restrict name)
{
	struct options *opts = &EGO->options;

	const size_t sz = sizeof(cl_double8);
	const size_t n  = opts->h_rays * opts->w_rays;

	void *h = clEnqueueMapBuffer(EGO->ocl->queue[0], EGO->data,
	                             CL_TRUE, CL_MAP_READ, 0, sz * n,
	                             0, NULL, NULL, NULL);

	FILE *f = fopen(name, "wb");
	fwrite(&sz,           sizeof(size_t), 1, f);
	fwrite(&opts->w_rays, sizeof(size_t), 1, f);
	fwrite(&opts->h_rays, sizeof(size_t), 1, f);
	fwrite( h,            sz,             n, f);
	fclose(f);

	clEnqueueUnmapMemObject(EGO->ocl->queue[0], EGO->data,
	                        h, 0, NULL, NULL);
}
