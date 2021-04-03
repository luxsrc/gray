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

static inline size_t
max(size_t a, size_t b)
{
	return a > b ? a : b;
}

real
evolve(Lux_job *ego, real t, real target, size_t n_sub)
{
	Lux_opencl        *ocl    = EGO->ocl;
	Lux_opencl_kernel *evolve = EGO->evolve;

	struct param *p = &EGO->param;
	struct setup *s = &EGO->setup;

	const  size_t sz     = s->precision;
	const  size_t n_data = EGO->n_coor + EGO->n_freq * 2;
	const  size_t n_info = EGO->n_info;

	const  size_t shape[] = {p->h_rays, p->w_rays};

	size_t arg_num = 0;

	evolve->setM(evolve, arg_num, EGO->data);
	arg_num++;
	evolve->setM(evolve, arg_num, EGO->info);
	arg_num++;
	evolve->setR(evolve, arg_num, target - t);
	arg_num++;
	evolve->setW(evolve, arg_num, n_sub);
	arg_num++;
	evolve->setS(evolve, arg_num, sz * max(n_data, n_info));
	arg_num++;
	evolve->set(evolve, arg_num, sizeof(cl_float8), &(EGO->bounding_box));
	arg_num++;
	evolve->set(evolve, arg_num, sizeof(cl_int4), &(EGO->num_points));
	arg_num++;
	/* We have 40 Gammas + 10 metric components + 1 fluid property at t1 */
	for (size_t old_arg_num = arg_num; arg_num < old_arg_num + 51; arg_num++)
		evolve->setM(evolve, arg_num, EGO->spacetime_t1[arg_num-old_arg_num]);
	/* And here the 40 Gammas + 10 metric components + 1 fluid property at t2 */
	for (size_t old_arg_num = arg_num; arg_num < old_arg_num + 51; arg_num++)
		evolve->setM(evolve, arg_num, EGO->spacetime_t2[arg_num-old_arg_num]);

	return ocl->exec(ocl, evolve, 2, shape);
}
