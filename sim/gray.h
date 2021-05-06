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
 **
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
#include <lux/numeric.h>
#include <lux/opencl.h>
#include <lux/strutils.h>

#include "icond.h"
#include "param.h"
#include "setup.h"

/** Max number of times in the HDF5 file and max length of the group
 ** name in the files */

#define MAX_AVAILABLE_TIMES 1024
#define MAX_TIME_NAME_LENGTH 64

/* You cannot simply increase this, you also have to add more horizons
 * in the struct and pass them to the kernel.  */
#define MAX_NUM_HORIZONS 3

/**
 ** Run-time data structure for GRay2
 **
 ** To take advantage of all the low level features provided by lux,
 ** GRay2 is implemented as a lux module.  Its runtime data is stored
 ** in a subclass of Lux_job so that it can be loaded by the lux
 ** runtime.
 **/
struct gray {
	Lux_job super;

	struct icond icond;
	struct param param;
	struct setup setup;

	size_t n_coor;
	size_t n_freq;
	size_t n_info;

	Lux_opencl *ocl;
	cl_mem data;
	cl_mem info;
	Lux_opencl_kernel *evolve;

	/* Grid details */
	/* Bounding_box is a vector with 8 numbers:
	 * {tmin, xmin, ymin, zmin, tmax, xmax, ymax, zmax} */
	/* tmin and tmax are between the two lodaded timesteps */

	/* We need these quantities to convert from unnormalized OpenCL coordiantes
	   to physical coordiantes and viceversa. */
	cl_float8 bounding_box;
	/* Points along the various coordinates */
	cl_int4 num_points;			/* The .s0 coordinate is not used */
	/* num_points.s1 contains point along the x direction */
	/* num_points.s2 contains point along the y direction */
	/* num_points.s3 contains point along the z direction */

	/* We need 40+10+1 == 51 images to contain all the 40 Christoffel symbols,
	   10 metric components, and 1 fluid variable */

	/* We always have two timesteps loaded */
	cl_mem spacetime_t1[40+10+5];
	cl_mem spacetime_t2[40+10+5];

	char available_times[MAX_AVAILABLE_TIMES][MAX_TIME_NAME_LENGTH];

	cl_float max_available_time;

	/* If horizon_is_valid is positive, then we are going to process
	 * this horizon. */
	cl_int    ah_valid_1[MAX_AVAILABLE_TIMES];
	cl_float4 ah_centr_1[MAX_AVAILABLE_TIMES];
	/* Minimum and maximum radii */
	cl_float  ah_min_r_1[MAX_AVAILABLE_TIMES];
	cl_float  ah_max_r_1[MAX_AVAILABLE_TIMES];

	cl_int    ah_valid_2[MAX_AVAILABLE_TIMES];
	cl_float4 ah_centr_2[MAX_AVAILABLE_TIMES];
	cl_float  ah_min_r_2[MAX_AVAILABLE_TIMES];
	cl_float  ah_max_r_2[MAX_AVAILABLE_TIMES];

	cl_int    ah_valid_3[MAX_AVAILABLE_TIMES];
	cl_float4 ah_centr_3[MAX_AVAILABLE_TIMES];
	cl_float  ah_min_r_3[MAX_AVAILABLE_TIMES];
	cl_float  ah_max_r_3[MAX_AVAILABLE_TIMES];

};


#define EGO ((struct gray *)ego)
#define CKR lux_check_func_success

/** Build the OpenCL module for GRay2 */
extern Lux_opencl *build(Lux_job *);

/** Set the initial conditions */
extern void icond(Lux_job *, real);

/** Evolve the states of photons to the next (super) step */
extern real evolve(Lux_job *, real, real, size_t, size_t snapshot_number);

/** Output data to a file */
extern void dump(Lux_job *, size_t);

/** I/O helper functions */
extern size_t populate_ego_available_times(Lux_job *);
extern size_t load_coordinates(Lux_job *);
extern size_t load_horizons(Lux_job *);
extern size_t load_snapshot(Lux_job *, size_t, size_t);
extern void   copy_snapshot(Lux_job *, size_t);

#endif /* _GRAY_H */
