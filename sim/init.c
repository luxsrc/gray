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

/** \page newkern New OpenCL Kernels
 **
 ** Extend GRay2 by adding new OpenCL kernels
 **
 ** GRay2 uses the just-in-time compilation feature of OpenCL to build
 ** computation kernels at run-time.  Most of the low level OpenCL
 ** codes are actually in a lux module called "opencl".  GRay2
 ** developers simply need to load this module with a list of OpenCL
 ** source codes, e.g.,
 ** \code{.c}
 **   const char *buf   = "static constant real a_spin = 0.999;\n";
 **   const char *src[] = {buf, "sim/KS.cl", "sim/RK4.cl", "sim/AoS.cl", NULL};
 **   const char *flags = "-cl-mad-enable";
 **   struct LuxOopencl otps = {..., flags, src};
 **   Lux_opencl *ocl = lux_load("opencl", &opts);
 ** \endcode
 ** and then an OpenCL kernel can be obtained and run by
 ** \code{.c}
 **   cl_kernel icond = ocl->mkkern(ocl, "icond_drv");
 **   ...
 **   clEnqueueNDRangeKernel(ocl->que, icond, ...);
 ** \endcode
 ** Therefore, with this powerful lux module, it is straightforward to
 ** add a new OpenCL kernels to GRay2:
 **
 ** -# Name the OpenCL source code with an extension ".cl" and add it
 **    to the "sim/" source code folder.
 ** -# In "sim/Makefile.am", append the new file name to dist_krn_DATA.
 ** -# Add new code to the C files in "sim" to use the new kernel, or
 **    make the new source code default in "sim/gray.c" if necessary.
 **
 ** Note that, however, the developer is responsible to make sure that
 ** the new OpenCL source code is compatible with other OpenCL codes.
 ** This is because GRay2 place all the OpenCL codes together and
 ** build them as a single program.
 **/
#include "gray.h"
#include <stdio.h>

int
init(Lux_job *ego)
{
	struct icond *i = &EGO->icond;
	struct param *p = &EGO->param;
	struct setup *s = &EGO->setup;

	const size_t sz      =  s->precision;
	const size_t n_vars  =  p->n_freq * 2 + 8;
	const size_t n_rays  =  p->h_rays * p->w_rays;
	const size_t shape[] = {p->h_rays,  p->w_rays};

	char buf[1024];
	const char *src[] = {buf, "KS.cl", "RK4.cl", "AoS.cl", NULL};
	struct LuxOopencl opts = OPENCL_NULL;

	Lux_opencl        *ocl;
	Lux_opencl_kernel *icond, *evol;
	cl_mem diag, data;

	lux_debug("GRay2: initializing job %p\n", ego);

	snprintf(buf, sizeof(buf),
	         "__constant real   a_spin = %g;\n"
	         "__constant size_t w_rays = %zu;\n"
	         "__constant size_t h_rays = %zu;\n"
	         "__constant size_t n_vars = %zu;\n",
	         p->a_spin,
	         p->w_rays,
	         p->h_rays,
	         n_vars);

	opts.base    = init;
	opts.iplf    = s->i_platform;
	opts.idev    = s->i_device;
	opts.devtype = s->device_type;
	opts.realsz  = s->precision;
	opts.src     = src;

	/* Load the OpenCL module with opts */
	CKR(ocl = lux_load("opencl", &opts), cleanup1);

	/* Allocate diagnostic and states/data buffers */
	CKR(diag = ocl->mk(ocl, CL_MEM_READ_WRITE, sz * n_rays),          cleanup2);
	CKR(data = ocl->mk(ocl, CL_MEM_READ_WRITE, sz * n_rays * n_vars), cleanup3);

	/* Create the "init" kernel: use EGO->bsz_max as a tmp variable */
	CKR(icond = ocl->mkkern(ocl, "icond_drv"), cleanup4);

	/* Initialize the states buffer */
	/** \todo check errors */
	ocl->setM(ocl, icond, 0, diag);
	ocl->setM(ocl, icond, 1, data);
	ocl->setR(ocl, icond, 2, i->w_img);
	ocl->setR(ocl, icond, 3, i->h_img);
	ocl->setR(ocl, icond, 4, i->r_obs);
	ocl->setR(ocl, icond, 5, i->i_obs);
	ocl->setR(ocl, icond, 6, i->j_obs);
	ocl->exec(ocl, icond, 2, shape);
	ocl->rmkern(ocl, icond);

	/* Create the "evol" kernel: save EGO->bsz_max for ego->exec() */
	CKR(evol = ocl->mkkern(ocl, "integrate_drv"), cleanup4);

	EGO->ocl  = ocl;
	EGO->diag = diag;
	EGO->data = data;
	EGO->evol = evol;
	return EXIT_SUCCESS;

 cleanup4:
	ocl->rm(ocl, data);
 cleanup3:
	ocl->rm(ocl, diag);
 cleanup2:
	lux_unload(ocl);
 cleanup1:
	return EXIT_FAILURE;
}
