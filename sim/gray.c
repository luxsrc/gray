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

#include <lux/mangle.h>
#include <lux/zalloc.h>

#include <hdf5.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

static int
_conf(Lux_job *ego, const char *restrict arg)
{
	/** \page newopts New Run-Time Options
	 **
	 ** Turn hard-wired constants into run-time options
	 **
	 ** GRay2 uses the lux framework and hence follows lux's
	 ** approach to support many run-time options.  To turn
	 ** hard-wired constants into run-time options, one needs to
	 **
	 ** -# Add an option table ".opts" file in the "sim/"
	 **    directory.
	 ** -# Embed the automatically generated structure to `struct
	 **    gray` in "sim/gray.h"
	 ** -# Logically `&&` the automatically generated configure
	 **    function to the return values of `_conf()` in
	 **    "sim/gray.c".
	 **/
	lux_debug("GRay2: configuring instance %p with \"%s\"\n", ego, arg);

	return icond_config(&EGO->icond, arg) &&
	       param_config(&EGO->param, arg) &&
	       setup_config(&EGO->setup, arg);
}

static int
_init(Lux_job *ego)
{
	Lux_opencl *ocl; /* to be loaded */

	struct param *p = &EGO->param;
	struct setup *s = &EGO->setup;

	const size_t sz     = s->precision;
	const size_t n_rays = p->h_rays * p->w_rays;
	const size_t n_data = EGO->n_coor + p->n_freq * 2;
	const size_t n_info = EGO->n_info;

	cl_mem_flags flags  = CL_MEM_READ_WRITE;

	lux_debug("GRay2: initializing instance %p\n", ego);

	CKR(EGO->ocl    = ocl    = build(ego),                       cleanup1);
	CKR(EGO->data   = ocl->mk(ocl, sz * n_rays * n_data, flags), cleanup2);
	CKR(EGO->info   = ocl->mk(ocl, sz * n_rays * n_info, flags), cleanup3);
	CKR(EGO->evolve = ocl->mkkern(ocl, "evolve_drv"),            cleanup4);

	return EXIT_SUCCESS;

 cleanup4:
	ocl->rm(ocl, EGO->info);
 cleanup3:
	ocl->rm(ocl, EGO->data);
 cleanup2:
	lux_unload(EGO->ocl);
 cleanup1:
	return EXIT_FAILURE;
}

static int
_exec(Lux_job *ego)
{
	struct param *p = &EGO->param;

	const  size_t n_rays = p->h_rays * p->w_rays;
	const  double dt     = -1.0;
	const  size_t n_sub  = 1024;

	size_t i;

	lux_debug("GRay2: executing instance %p\n", ego);

	lux_print("GRay2: Reading spacetime from file %s\n", p->dyst_file);

	/* We perform basic checks here */
	lux_check_failure_code(access(p->dyst_file, F_OK), cleanup1);
	hid_t file_id = H5Fopen(p->dyst_file, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file_id == -1) goto cleanup2;

	/* We list all the available times in the file */
	lux_check_failure_code(populate_ego_available_times(ego), cleanup3);


	/* We load the coordinates */
	lux_check_failure_code(load_spatial_bounding_box(ego), cleanup3);

	/* Here we read the snapshot at t = tmin */
	lux_check_failure_code(load_next_snapshot(ego, 0), cleanup3);

	icond(ego);
	dump(ego, 0);

	char *rem;
	size_t spacetime_frozen = 0;
	size_t time_snapshot_index;

	double current_time = strtod(EGO->available_times[0], &rem);
	double t2;

	/* We check whether we have more than one times. */
	if (EGO->max_available_time == current_time){
		lux_print("Found only one time in data, freezing spacetime\n");
		spacetime_frozen = 1;
		/* We are going to set t2 = current_time and the kernel will know
		 * what to do. */
		t2 = current_time;
		/* We have to fill t2 with something. */
		copy_snapshot_to_t2(ego);
	}else{
		lux_print("Max time in data: %f\n", EGO->max_available_time);
		spacetime_frozen = 0;
		time_snapshot_index = 1;
		lux_check_failure_code(load_next_snapshot(ego, time_snapshot_index), cleanup3);
		t2 = strtod(EGO->available_times[time_snapshot_index], &rem);
	}

	EGO->bounding_box.s0 = current_time;
	EGO->bounding_box.s4 = t2;

	for(i = 0; i < 1; ++i) {
		double s;
		clock_t start, end;
		double cpu_time_used;
		lux_print("%4.1f -> %4.1f", current_time, current_time + dt);
		s = evolve(ego) * 1e-9;	/* seconds */
		start = clock();
		dump(ego, i+1);
		end = clock();
		cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		lux_print(": DONE (%.4e s", s);
		lux_print(", dumping: %.4e s)\n", cpu_time_used);
		current_time += dt;

		/* If we have a dynamic spacetime, we have to load the next snapshot */
		if (spacetime_frozen != 1){
			/* Here we shift to the new snapshot */
			/* We have to change EGO->spacetime_t1 and the bounding_box */
			if (current_time <= EGO->max_available_time){
				lux_print("Reached max available time");
				return EXIT_SUCCESS;
			}
			if (current_time <= t2){
				for (size_t j=0; j < 40; ++j)
					EGO->spacetime_t1[j] = EGO->spacetime_t2[j];
				EGO->bounding_box.s0 = t2;
				/* Now we read the new snapshot */
				time_snapshot_index++;
				lux_check_failure_code(load_next_snapshot(ego, time_snapshot_index), cleanup3);
				t2 = strtod(EGO->available_times[time_snapshot_index], &rem);
				EGO->bounding_box.s4 = t2;
			}
		}
	}

	return EXIT_SUCCESS;

cleanup1:
	lux_print("ERROR: File %s could not be read\n", p->dyst_file);
	return EXIT_FAILURE;
cleanup2:
	lux_print("ERROR: File %s is not a valid HDF5 file\n", p->dyst_file);
	return EXIT_FAILURE;
cleanup3:
	return EXIT_FAILURE;
}

void *
LUX_MKMOD(const void *opts)
{
	void *ego;

	lux_debug("GRay2: constructing with options %p\n", opts);

	ego = zalloc(sizeof(struct gray));
	if(ego) {
		EGO->super.conf = _conf;
		EGO->super.init = _init;
		EGO->super.exec = _exec;
		icond_init(&EGO->icond);
		param_init(&EGO->param);
		setup_init(&EGO->setup);
		EGO->n_coor = 8; /* \todo adjust using setup.coordinates */
		EGO->n_info = 1; /* \todo adjust using setup.coordinates */
	}
	return ego;
}

void
LUX_RMMOD(void *ego)
{
	Lux_opencl *ocl = EGO->ocl;

	lux_debug("GRay2: destructing instance %p\n", ego);

	if(EGO->evolve)
		ocl->rmkern(ocl, EGO->evolve);
	if(EGO->info)
		ocl->rm(ocl, EGO->info);
	if(EGO->data)
		ocl->rm(ocl, EGO->data);
	if(EGO->ocl)
		lux_unload(EGO->ocl);
	free(ego);
}
