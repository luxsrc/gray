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
min(size_t a, size_t b)
{
	return a < b ? a : b;
}

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

	const size_t n_data  = EGO->n_coor + EGO->n_freq * 2;
	const size_t n_info  = 1;
	const size_t e_chunk = min(16, n_data & ~(n_data-1)); /* number of real elements in chunk */
	const size_t n_chunk = n_data / e_chunk;              /* number of chunks */

	char buf[1024];
	const char *src[] = {buf,
	                     "preamble.cl",
	                     p->coordinates,
	                     "flow.cl",
	                     "rt.cl",
	                     s->morder,
	                     "phys.cl",
	                     s->scheme,
	                     "driver.cl",
	                     NULL};

	snprintf(buf, sizeof(buf),
	         "#define a_spin K(%.18f)\n" /* DBL_EPSILON ~ 1e-16 */
	         "#define n_freq %zu\n"
	         "#define n_data %zu\n"
	         "#define n_info %zu\n"
	         "#define n_rays %zu\n"
	         "#define w_rays %zu\n"
	         "#define h_rays %zu\n"
	         "#define n_chunk %zu\n"
	         "typedef real%zu realE;\n",
	         p->a_spin,
	         EGO->n_freq,
	         n_data,
	         n_info,
	         p->h_rays * p->w_rays,
	         p->w_rays,
	         p->h_rays,
	         n_chunk,
	         e_chunk);

	lux_print("n_data  = %zu\n"
	          "n_info  = %zu\n"
	          "e_chunk = %zu\n",
	          n_data,
	          n_info,
	          e_chunk);

	opts.base    = build; /* this function */
	opts.iplf    = s->i_platform;
	opts.idev    = s->i_device;
	opts.devtype = s->device_type;
	opts.realsz  = s->precision;
	opts.flags   = s->kflags;
	opts.src     = src;

	return lux_load("opencl", &opts);
}
