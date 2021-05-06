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


	size_t arg_num = 0;

	icond->setM(icond, arg_num, EGO->data);
	arg_num++;
	icond->setM(icond, arg_num, EGO->info);
	arg_num++;

	icond->setR(icond, arg_num, i->w_img);
	arg_num++;
	icond->setR(icond, arg_num, i->h_img);
	arg_num++;
	icond->setR(icond, arg_num, i->r_obs);
	arg_num++;
	icond->setR(icond, arg_num, i->i_obs);
	arg_num++;
	icond->setR(icond, arg_num, i->j_obs);
	arg_num++;

	icond->setS(icond, arg_num, sz * max(n_data, n_info));
	arg_num++;

	icond->set(icond, arg_num, sizeof(cl_float8), &(EGO->bounding_box));
	arg_num++;
	icond->set(icond, arg_num, sizeof(cl_int4), &(EGO->num_points));
	arg_num++;
	/* We have 40 Gammas + 10 metric components + 5 fluid property at t1 */
	for (size_t old_arg_num = arg_num; arg_num < old_arg_num + 55; arg_num++)
		icond->setM(icond, arg_num, EGO->spacetime_t1[arg_num-old_arg_num]);
	/* And here the 40 Gammas + 10 metric components + 5 fluid property at t2 */
	for (size_t old_arg_num = arg_num; arg_num < old_arg_num + 55; arg_num++)
		icond->setM(icond, arg_num, EGO->spacetime_t2[arg_num-old_arg_num]);

	(void)ocl->exec(ocl, icond, 2, shape);

	ocl->rmkern(ocl, icond);
}
