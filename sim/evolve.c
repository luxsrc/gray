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

double
evolve(Lux_job *ego)
{
	Lux_opencl   *ocl     =  EGO->ocl;
	struct param *p       = &EGO->param;
	const  size_t shape[] = {p->h_rays,  p->w_rays};

	double dt    = -1.0;
	size_t n_sub = 1024;

	ocl->setM(ocl, EGO->evolve, 0, EGO->data);
	ocl->setM(ocl, EGO->evolve, 1, EGO->info);
	ocl->setR(ocl, EGO->evolve, 2, dt);
	ocl->setW(ocl, EGO->evolve, 3, n_sub);

	return ocl->exec(ocl, EGO->evolve, 2, shape);
}
