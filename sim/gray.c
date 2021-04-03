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

#include <math.h>
#include <stdio.h>
#include <unistd.h>				/* For access and F_OK */
#include <hdf5.h>

inline static real time_at_snapshot(Lux_job *ego, int snapshot_number){
	/* Return the time corresponding to the given snapshot */

	/* Times are saved as chars, so we need to do operations with this
	 * data type.  This is used to read the times in the HDF5 files. */
	char *rem;

	return strtod(EGO->available_times[snapshot_number], &rem);
}


static void find_snapshot(Lux_job *ego, real t, size_t *snap){
	/* Find snapshot number so that t1 <= t <= t2, where t1 is the time
	 * corresponding to the snapshot number */

	real t1, t2;

	/* We assume that snapshots they are ordered from the min to the max. */
	*snap = -1;
	/* We have already performed all the necessary checks, so this loop should
	 * be well defined. */
	do{
		(*snap)++;
		t1 = time_at_snapshot(ego, *snap);
		t2 = time_at_snapshot(ego, *snap + 1);
		/* It has to be that slow_light_t2 > slow_light_t1 */
	}while(!(t >= t1 && t <= t2));
}


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
	int invalid;
	real *nu;

	lux_debug("GRay2: configuring instance %p with \"%s\"\n", ego, arg);

	nu = EGO->param.nu; /* save the previous nu */

	invalid = (icond_config(&EGO->icond, arg) &&
	           param_config(&EGO->param, arg) &&
	           setup_config(&EGO->setup, arg));

	if(EGO->param.nu != nu) { /* nu was configured */
		if(nu)
			free(nu); /* avoid memory leackage by freeing the old nu */

		nu = EGO->param.nu;
		if(isnan(nu[0])) {
			lux_print("nu: []\n");
			EGO->n_freq = 0;
		} else {
			size_t n;
			lux_print("nu: [%f", nu[0]);
			for(n = 1; !isnan(nu[n]); ++n)
				lux_print(", %f", nu[n]);
			lux_print("]\n");
			EGO->n_freq = n;
		}
	}

	return invalid;
}

static int
_init(Lux_job *ego)
{
	Lux_opencl *ocl; /* to be loaded */

	struct param *p = &EGO->param;
	struct setup *s = &EGO->setup;

	const size_t sz     = s->precision;
	const size_t n_rays = p->h_rays * p->w_rays;
	const size_t n_data = EGO->n_coor + EGO->n_freq * 2;
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
	struct setup *s = &EGO->setup;

	const  size_t n_rays  = p->h_rays * p->w_rays;

	       size_t i       = s->i_init;
	const  size_t n_sub   = s->n_sub;
	const  size_t n_dump  = s->n_dump;

	const  real t_init  = s->t_init;
	const  real dt_dump = s->dt_dump;
	/* If we are working with slow light, these are the two extrema. */
	real slow_light_t1, slow_light_t2;

	size_t frozen_spacetime = p->enable_fast_light;
	size_t only_one_snapshot = 0;

	lux_debug("GRay2: executing instance %p\n", ego);

	lux_print("GRay2: Reading spacetime from file %s\n", p->dyst_file);

	/* We perform basic checks here */
	lux_check_failure_code(access(p->dyst_file, F_OK), cleanup1);
	hid_t file_id = H5Fopen(p->dyst_file, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file_id == -1) goto cleanup2;

	/* We list all the available times in the file */
	lux_check_failure_code(populate_ego_available_times(ego), cleanup3);

	/* We load the coordinates */
	lux_check_failure_code(load_coordinates(ego), cleanup3);

	/* If max_available_time is equal to the first time available, it
	 * means that it is the only one. */

	real min_available_time = time_at_snapshot(ego, 0);
	if (EGO->max_available_time == min_available_time){
		lux_print("Found only one time in data, freezing spacetime\n");
		only_one_snapshot = 1;
		frozen_spacetime = 1;
	}else{
		/* It does not make sense to perform the integration if we don't have
		 * the desired initial time and final in range, unless we only have one
		 * time snapshot. */
		if ((t_init < min_available_time || t_init > EGO->max_available_time)){
			lux_print("ERROR: t_init (%4.1f) is outside domain of the data (%5.1f, %5.1f)\n",
					  t_init, min_available_time, EGO->max_available_time);
			return EXIT_FAILURE;
		}
		real t_final = t_init + (i+1) * dt_dump * n_dump;
		if ((t_final < min_available_time || t_final > EGO->max_available_time)){
			lux_print("ERROR: t_final (%4.1f) is outside domain of the data (%5.1f, %5.1f)\n",
					  t_final, min_available_time, EGO->max_available_time);
			return EXIT_FAILURE;
		}
	}

	/* Snapshot of interest */
	size_t snap_number;

	if (frozen_spacetime){
		lux_print("Assuming fast light\n");
		if (only_one_snapshot){
			/* 1 here means "load in t1" */
			lux_check_failure_code(load_snapshot(ego, 0, 1), cleanup3);
		}else{
             /* Here we read the snapshot at t1 and t2 so that they contain t_init. */
			find_snapshot(ego, t_init, &snap_number);
			/* 1 here means "load in t1" */
			lux_check_failure_code(load_snapshot(ego, snap_number, 1), cleanup3);
		}
		/* We have to fill t2 with something, otherwise it will produce errors.
		 * We fill with the same data as t1.  Here 0 means "to_t2" */
		copy_snapshot(ego, 0);
		/* Next, we disable time interpolation by setting the two time extrema
		 * of the bounding box to be the same */
		EGO->bounding_box.s0 = 0;
		EGO->bounding_box.s4 = 0;
	}else{
		lux_print("Working with slow light\n");
		/* Here we read the snapshot at t1 and t2 so that they contain t_init. */
		find_snapshot(ego, t_init, &snap_number);
		slow_light_t1 = time_at_snapshot(ego, snap_number);
		slow_light_t2 = time_at_snapshot(ego, snap_number + 1);
		/* 1 here means "load in t1" */
		lux_check_failure_code(load_snapshot(ego, snap_number, 1), cleanup3);
		/* 0 here means "load in t2" */
		lux_check_failure_code(load_snapshot(ego, snap_number + 1, 0), cleanup3);
		EGO->bounding_box.s0 = slow_light_t1;
		EGO->bounding_box.s4 = slow_light_t2;
	}

	lux_print("%zu:  initialize at %4.1f", i, t_init);
	icond(ego, t_init);
	dump (ego, i);
	lux_print(": DONE\n");

	while(i < n_dump) {
		real ns, t, target;

		t      = t_init +    i  * dt_dump;
		target = t_init + (++i) * dt_dump;

		lux_print("%zu: %4.1f -> %4.1f", i, t, target);
		ns = evolve(ego, t, target, n_sub);
		dump(ego, i);
		lux_print(": DONE (%.3gns/step/ray)\n", ns/n_sub/n_rays);

		/* If we are not freezing the spacetime, we need to change the snapshots */
		if (!frozen_spacetime && (target < slow_light_t1 || target > slow_light_t2)){

			/* If snap_number is off by 1 compared to old_snap_number, this
			 * means that we can read only one of the two snapshots and copy
			 * over the other one.  If it is off by more than 1, then we have to
			 * read them both. */
			size_t old_snap_number = snap_number;
			find_snapshot(ego, target, &snap_number);
			slow_light_t1 = time_at_snapshot(ego, snap_number);
			slow_light_t2 = time_at_snapshot(ego, snap_number + 1);

			if (snap_number == old_snap_number + 1){
				/* In this case, the old t2 has to become the new t1.  Here 1
				 * means "copy to t1" */
				copy_snapshot(ego, 1);
				/* 0 here means "load in t2" */
				lux_check_failure_code(load_snapshot(ego, snap_number + 1, 0), cleanup3);
			}else if (snap_number == old_snap_number - 1){
				/* In this case, the old t1 has to become the new t2.  Here 0
				 * means "copy to t2" */
				copy_snapshot(ego, 0);
				/* 1 here means "load in t1" */
				lux_check_failure_code(load_snapshot(ego, snap_number, 1), cleanup3);
			}else{
				/* We have to read them both */
				/* 1 here means "load in t1" */
				lux_check_failure_code(load_snapshot(ego, snap_number, 1), cleanup3);
				/* 0 here means "load in t2" */
				lux_check_failure_code(load_snapshot(ego, snap_number + 1, 0), cleanup3);
			}
			/* Update bounding box */
			EGO->bounding_box.s0 = slow_light_t1;
			EGO->bounding_box.s4 = slow_light_t2;
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
		EGO->n_coor = 8; /** \todo Adjust n_coor using setup.coordinates. */
		EGO->n_freq = 0;
		EGO->n_info = 1; /** \todo Adjust n_info using setup.coordinates. */
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
