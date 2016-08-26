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
#include <stdio.h>

/** \todo Implement load() */

void
dump(Lux_job *ego, size_t i)
{
	Lux_opencl *ocl = EGO->ocl;

	struct param *p = &EGO->param;
	struct setup *s = &EGO->setup;

	const  size_t sz     = s->precision;
	const  size_t n_data = EGO->n_coor + EGO->n_freq * 2;
	const  size_t n_info = EGO->n_info;
	const  size_t n_rays = p->h_rays * p->w_rays;

	void *data = ocl->mmap(ocl, EGO->data, sz * n_rays * n_data);
	void *info = ocl->mmap(ocl, EGO->info, sz * n_rays * n_info);

	char  buf[64];
	FILE *f;

	snprintf(buf, sizeof(buf), s->outfile, i);
	f = fopen(buf, "wb");

	fwrite(&sz,        sizeof(size_t), 1,      f);
	fwrite(&n_data,    sizeof(size_t), 1,      f);
	fwrite(&p->w_rays, sizeof(size_t), 1,      f);
	fwrite(&p->h_rays, sizeof(size_t), 1,      f);
	fwrite( data,      sz * n_data,    n_rays, f);
	fwrite( info,      sz * n_info,    n_rays, f);

	fclose(f);

	ocl->munmap(ocl, EGO->info, info);
	ocl->munmap(ocl, EGO->data, data);
}
