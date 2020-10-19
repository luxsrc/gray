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

double
evolve(Lux_job *ego)
{
	Lux_opencl *ocl = EGO->ocl;

	struct param *p = &EGO->param;
	struct setup *s = &EGO->setup;

	const  size_t sz     = s->precision;
	const  size_t n_data = EGO->n_coor + p->n_freq * 2;
	const  size_t n_info = EGO->n_info;

	const double dt      = 1.0;
	const size_t n_sub   = 1024;
	const size_t shape[] = {p->h_rays, p->w_rays};

	size_t arg_num = 0;

  ocl->setM(ocl, EGO->evolve, arg_num, EGO->data);
  arg_num++;
  ocl->setM(ocl, EGO->evolve, arg_num, EGO->info);
  arg_num++;
  ocl->setR(ocl, EGO->evolve, arg_num, dt);
  arg_num++;
  ocl->setW(ocl, EGO->evolve, arg_num, n_sub);
  arg_num++;
  ocl->setS(ocl, EGO->evolve, arg_num, sz * max(n_data, n_info));
  arg_num++;
  ocl->set(ocl, EGO->evolve, arg_num, sizeof(cl_float8), &(EGO->bounding_box));
  arg_num++;
  /* We have 40 Gammas */
  for (size_t old_arg_num = arg_num; arg_num < old_arg_num + 40; arg_num++)
	  ocl->setM(ocl, EGO->evolve, arg_num, EGO->spacetime[arg_num-old_arg_num]);

  return ocl->exec(ocl, EGO->evolve, 2, shape);
}
