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

Lux_opencl *
build(Lux_job *ego)
{
	/** \page newkern New OpenCL Kernels
	 **
	 ** Extend GRay2 by adding new OpenCL kernels
	 **
	 ** GRay2 uses the just-in-time compilation feature of OpenCL
	 ** to build computation kernels at run-time.  Most of the low
	 ** level OpenCL codes are actually in a lux module called
	 ** "opencl".  GRay2 developers simply need to load this
	 ** module with a list of OpenCL source codes, e.g.,
	 ** \code{.c}
	 **   const char *buf   = "static constant real a_spin = 0.999;\n";
	 **   const char *src[] = {buf, "KS", "RK4", "AoS", NULL};
	 **   const char *flags = "-cl-mad-enable";
	 **   struct LuxOopencl otps = {..., flags, src};
	 **   Lux_opencl *ocl = lux_load("opencl", &opts);
	 ** \endcode
	 ** and then an OpenCL kernel can be obtained and run by
	 ** \code{.c}
	 **   Lux_opencl_kernel *icond = ocl->mkkern(ocl, "icond_drv");
	 **   ...
	 **   ocl->exec(ocl, icond, ...);
	 ** \endcode
	 ** Therefore, with this powerful lux module, it is
	 ** straightforward to add a new OpenCL kernels to GRay2:
	 **
	 ** -# Name the OpenCL source code with an extension ".cl" and
	 **    add it to the "sim/" source code folder.
	 ** -# In "sim/Makefile.am", append the new file name to
               dist_krn_DATA.
	 ** -# Add new code to the C files in "sim" to use the new
	 **    kernel, or make the new source code default in
	 **    "sim/gray.c" if necessary.
	 **
	 ** Note that, however, the developer is responsible to make
	 ** sure that the new OpenCL source code is compatible with
	 ** other OpenCL codes.  This is because GRay2 place all the
	 ** OpenCL codes together and build them as a single program.
	 **/
	struct LuxOopencl opts = OPENCL_NULL;

	struct param *p = &EGO->param;
	struct setup *s = &EGO->setup;

	char buf[1024];
	const char *src[] = {buf,
	                     p->coordinates,
	                     s->scheme,
	                     s->morder,
	                     "driver.cl",
	                     NULL};

	snprintf(buf, sizeof(buf),
	         "__constant real   a_spin = %g;\n"
	         "__constant size_t w_rays = %zu;\n"
	         "__constant size_t h_rays = %zu;\n"
	         "__constant size_t n_rays = %zu;\n"
	         "__constant size_t n_vars = %zu;\n",
	         p->a_spin,
	         p->w_rays,
	         p->h_rays,
	         p->h_rays * p->w_rays,
	         p->n_freq * 2 + 8);

	opts.base    = build; /* this function */
	opts.iplf    = s->i_platform;
	opts.idev    = s->i_device;
	opts.devtype = s->device_type;
	opts.realsz  = s->precision;
	opts.flags   = s->kflags;
	opts.src     = src;

	return lux_load("opencl", &opts);
}
