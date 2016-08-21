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
	struct param *p = &EGO->param;

	const size_t sz = sizeof(cl_double8);
	const size_t n  = p->h_rays * p->w_rays;

	void *d = EGO->ocl->mmap(EGO->ocl, EGO->diag, sizeof(double) * n);
	void *h = EGO->ocl->mmap(EGO->ocl, EGO->data, sz * n);

	FILE *f = fopen(name, "wb");
	fwrite(&sz,        sizeof(size_t), 1, f);
	fwrite(&p->w_rays, sizeof(size_t), 1, f);
	fwrite(&p->h_rays, sizeof(size_t), 1, f);
	fwrite( h,         sz,             n, f);
	fwrite( d,         sizeof(double), n, f);
	fclose(f);

	EGO->ocl->munmap(EGO->ocl, EGO->diag, d);
	EGO->ocl->munmap(EGO->ocl, EGO->data, h);
}
