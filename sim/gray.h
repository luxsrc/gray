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
#ifndef _GRAY_H_
#define _GRAY_H_

#include <lux/check.h>
#include <lux/job.h>
#include <lux/numeric.h>
#include <lux/opencl.h>
#include "options.h"

struct gray {
	Lux_job        super;
	struct options options;
	Lux_opencl    *ocl;
	cl_mem         data;
	cl_kernel      evol;
};

#define EGO ((struct gray *)ego)
#define CKR lux_check_func_success

extern int  conf(Lux_job *, const char *);
extern int  init(Lux_job *);
extern int  exec(Lux_job *);
extern void dump(Lux_job *, const char *);

#endif /* _GRAY_H */
