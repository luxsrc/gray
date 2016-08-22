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

/** \file
 ** Data structure definitions and function declarations for GRay2
 **
 ** GRay2 is implemented as a lux module.  Its run-time data is stored
 ** in a subclass of Lux_job, which is defined in this header file.
 ** Additional structure that holds run-time adjustable parameters,
 ** constructor, destructor, internal functions, and standard methods
 ** in Lux_job, are all declared here as well.
 **/
#ifndef _GRAY_H_
#define _GRAY_H_

#include <lux.h>
#include <lux/check.h>
#include <lux/job.h>
#include <lux/mangle.h>
#include <lux/numeric.h>
#include <lux/opencl.h>

#include "icond.h"
#include "param.h"
#include "setup.h"

/**
 ** Run-time data for GRay2
 **
 ** To take advantage of all the low level features provided by lux,
 ** GRay2 is implemented as a lux module.  Its runtime data is stored
 ** in a subclass of Lux_job so that it can be loaded by the lux
 ** runtime.
 **/
struct gray {
	Lux_job      super;
	struct icond icond;
	struct param param;
	struct setup setup;
	Lux_opencl  *ocl;
	cl_mem       diag;
	cl_mem       data;
	cl_kernel    evol;
};

#define EGO ((struct gray *)ego)
#define CKR lux_check_func_success

void *LUX_MKMOD(const void *);
void  LUX_RMMOD(void *);

/** Configuration method in the Lux_Job interface */
extern int  conf(Lux_job *, const char *);

/** Initialization method in the Lux_Job interface */
extern int  init(Lux_job *);

/** Execution method in the Lux_Job interface */
extern int  exec(Lux_job *);

/** Output function */
extern void dump(Lux_job *, const char *);

#endif /* _GRAY_H */
