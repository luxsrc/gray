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

static inline size_t
max(size_t a, size_t b)
{
	return a > b ? a : b;
}

void
icond(Lux_job *ego, real t_init)
{
	Lux_opencl *ocl = EGO->ocl;

	struct icond *i = &EGO->icond;
	struct param *p = &EGO->param;
	struct setup *s = &EGO->setup;

	const  size_t sz     = s->precision;
	const  size_t n_data = EGO->n_coor + EGO->n_freq * 2;
	const  size_t n_info = EGO->n_info;

	const size_t shape[] = {p->h_rays, p->w_rays};

	Lux_opencl_kernel *icond;

	lux_debug("GRay2: executing job %p\n", ego);

	icond = ocl->mkkern(ocl, "icond_drv");

	(void)ocl->exec(ocl,
	                icond->with(icond,
	                            EGO->data,
	                            EGO->info,
	                            i->w_img,
	                            i->h_img,
	                            i->r_obs,
	                            i->i_obs,
	                            i->j_obs,
	                            sz * max(n_data, n_info)),
	                2, shape);

	ocl->rmkern(ocl, icond);
}
