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
 **   const char *buf   = "static constant double a_spin = 0.999;\n";
 **   const char *src[] = {buf, "sim/KS.cl", "sim/RK4.cl", "sim/AoS.cl", NULL};
 **   const char *flags = "-cl-mad-enable";
 **   struct LuxOopencl otps = {..., flags, src};
 **   Lux_opencl *ocl = lux_load("opencl", &opts);
 ** \endcode
 ** and then an OpenCL kernel can be obtained and run by
 ** \code{.c}
 **   cl_kernel init = ocl->mkkern(ocl, "init");
 **   ...
 **   clEnqueueNDRangeKernel(ocl->que, init, ...);
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

	size_t ray_sz = sizeof(real) * 8;
	size_t n_rays = p->h_rays * p->w_rays;
	const size_t gsz[] = {p->h_rays, p->w_rays};
	const size_t bsz[] = {1, 1};

	char buf[1024];
	const char *src[] = {buf, "KS.cl", "RK4.cl", "AoS.cl", NULL};
	struct LuxOopencl opts = {init, 0, 0, CL_DEVICE_TYPE_CPU, NULL, src};

	Lux_opencl *ocl;
	cl_mem      diag, data;
	cl_kernel   init, evol;

	lux_debug("GRay2: initializing job %p\n", ego);

	snprintf(buf, sizeof(buf),
	         "__constant double a_spin = %g;\n"
	         "__constant size_t w_rays = %zu;\n"
	         "__constant size_t h_rays = %zu;\n",
	         p->a_spin,
	         p->w_rays,
	         p->h_rays);

	CKR(ocl  = lux_load("opencl", &opts),                                       cleanup1);
	CKR(diag = ocl->mk(ocl->super, CL_MEM_READ_WRITE, sizeof(double) * n_rays), cleanup2);
	CKR(data = ocl->mk(ocl->super, CL_MEM_READ_WRITE, ray_sz         * n_rays), cleanup3);
	CKR(init = ocl->mkkern(ocl, "init"),                                        cleanup4);
	CKR(evol = ocl->mkkern(ocl, "evol"),                                        cleanup5);

	/** \todo check errors */
	ocl->set(ocl, init, 0, sizeof(cl_mem), &diag);
	ocl->set(ocl, init, 1, sizeof(cl_mem), &data);
	ocl->set(ocl, init, 2, sizeof(double), &i->w_img);
	ocl->set(ocl, init, 3, sizeof(double), &i->h_img);
	ocl->set(ocl, init, 4, sizeof(double), &i->r_obs);
	ocl->set(ocl, init, 5, sizeof(double), &i->i_obs);
	ocl->set(ocl, init, 6, sizeof(double), &i->j_obs);
	ocl->exec(ocl, init, 2, gsz, bsz);
	ocl->rmkern(init);

	EGO->ocl  = ocl;
	EGO->diag = diag;
	EGO->data = data;
	EGO->evol = evol;
	return EXIT_SUCCESS;

 cleanup5:
	ocl->rmkern(init);
 cleanup4:
	ocl->rm(data);
 cleanup3:
	ocl->rm(diag);
 cleanup2:
	lux_unload(ocl);
 cleanup1:
	return EXIT_FAILURE;
}
