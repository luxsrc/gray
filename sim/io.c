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
#include <hdf5.h>

/** \todo implement load() */

void
dump(Lux_job *ego, size_t i)
{
	Lux_opencl *ocl = EGO->ocl;

	struct param *p = &EGO->param;
	struct setup *s = &EGO->setup;

	const  size_t sz     = s->precision;
	const  size_t n_data = EGO->n_coor + p->n_freq * 2;
	const  size_t n_info = EGO->n_info;
	const  size_t n_rays = p->h_rays * p->w_rays;

	void *data = ocl->mmap(ocl, EGO->data, sz * n_rays * n_data);
	void *info = ocl->mmap(ocl, EGO->info, sz * n_rays * n_info);

	char  buf[64];
	FILE *f;

	snprintf(buf, sizeof(buf), s->outfile, i);
	f = fopen(buf, "wb");

	fwrite(&sz,        sizeof(size_t), 1,      f);
	fwrite(&n_data,    sizeof(size_t), 1,      f);
	fwrite(&p->w_rays, sizeof(size_t), 1,      f);
	fwrite(&p->h_rays, sizeof(size_t), 1,      f);
	fwrite( data,      sz * n_data,    n_rays, f);
	fwrite( info,      sz * n_info,    n_rays, f);

	fclose(f);

	ocl->munmap(ocl, EGO->info, info);
	ocl->munmap(ocl, EGO->data, data);
}

size_t
load_spacetime(Lux_job *ego, double time_double)
{

	struct param *p = &EGO->param;

	const char *file_name = p->dyst_file;
	const char *time_format = p->time_format;

	/* OpenCL Image properties */
	cl_image_format imgfmt;
	cl_image_desc imgdesc;
	cl_int err;

	/* HDF5 identifiers */
	hid_t file_id;
	hid_t group_id;

	/* The group name is a char, not a double */
	char time[64];

	snprintf(time, sizeof(time), time_format, time_double);

	file_id = H5Fopen(file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file_id == -1) {
		lux_print("ERROR: File %s is not a valid HDF5 file\n", file_name);
		return file_id;
	}
	/* Open group corresponding to time */
	group_id = H5Gopen(file_id, time, H5P_DEFAULT);
	if (group_id == -1) {
		lux_print("ERROR: Time %s not found\n", time);
		lux_print("Time has to be in the format specified by the option time_fmt ");
		lux_print("(currently, time_fmt = %s)\n", time_format);
		return group_id;
	}

	lux_debug("Reading time %s\n", time);

	return 0;

	/* FILE  *file; */

	/* size_t dim, size[DIM_MAX], count[DIM_MAX]; */
	/* void  *host; */

	/* cl_image_format imgfmt; */
	/* cl_image_desc   imgdesc; */
	/* cl_int err; */

	/* /\* Load spacetime data to host memory *\/ */
	/* file = fopen(name, "rb"); */
	/* fread(&dim,  sizeof(size_t), 1,   file); */
	/* fread(size,  sizeof(size_t), dim, file); */
	/* fread(count, sizeof(size_t), dim, file); */
	/* host = malloc(...); */
	/* fread(host, size, count, file); */
	/* fclose(file); */

	/* /\* Create "image" on device and copy spacetime data to it *\/ */
	/* imgfmt.image_channel_order     = CL_RGBA; /\* use four channels *\/ */
	/* imgfmt.image_channel_data_type = CL_FLAT; /\* each channel is a float *\/ */
	/* imgdesc.image_type = CL_MEM_OBJECT_IMAGE3D; */

	/* ego->spacetime = clCreateImage */
	/* 	(ego->ocl->super, */
	/* 	 CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, */
	/* 	 &image_format, */
	/* 	 &image_desc, */
	/* 	 host, */
	/* 	 &err); */

	/* free(host); */
}
