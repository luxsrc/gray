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
#include <time.h>

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
read_variable_from_h5_file_and_return_num_points(const hid_t group_id,
												 const char *var_name,
												 void **var_array)
{

	/* Here we read var_name from group_id and put in var_array*/
	/* var_array is a pointer to a pointer to the area of memory where the data
	 * will be written */
	/* We want a pointer to a pointer because we want to modify var_array
	 * with malloc */

	/* We allocate memory, so it has to be freed! */
	/* WARNING: The memory has to be freed! */

	/* The return value is the number of elements */

	herr_t status;
	hid_t datasetH5type;
	hid_t dataset_id, dataspace_id; /* identifiers for dsets*/

	lux_debug("Reading variable %s\n", var_name);

	dataset_id = H5Dopen(group_id, var_name, H5P_DEFAULT);
	if (dataset_id == -1) {
		lux_print("Error in opening dataset: %s", var_name);
		return dataset_id;
	}

	/* The dataspace will tell us about the size (in bytes) of the data */
	dataspace_id = H5Dget_space(dataset_id);
	if (dataspace_id == -1) {
		lux_print("Error in getting dataspace: %s", var_name);
		return dataset_id;
	}

	const size_t total_num_bytes = H5Dget_storage_size(dataset_id);

	/* Here we allocate the memory */
	/* IT MUST BE FREED! */
	*var_array = malloc(total_num_bytes);

	/* To read the data, we must know what type is it */
	datasetH5type = H5Tget_native_type(H5Dget_type(dataset_id), H5T_DIR_DEFAULT);
	if (datasetH5type == -1) {
		lux_print("Error in determining type in dataset: %s", var_name);
		return datasetH5type;
	}

	/* Size in bytes of each signle element */
	const size_t sz = H5Tget_size(datasetH5type);

	status = H5Dread(dataset_id, datasetH5type, H5S_ALL, H5S_ALL, H5P_DEFAULT,
					 *var_array);

	if (status != 0) {
		lux_print("Error in reading dataset: %s", var_name);
		return status;
	}

	status = H5Dclose(dataset_id);
	if (status != 0) {
		printf("Error in closing dataset: %s", var_name);
		return status;
	}

	status = H5Sclose(dataspace_id);
	if (status != 0) {
		printf("Error in closing dataspace: %s", var_name);
		return status;
	}

	return total_num_bytes / sz;
}

size_t
load_spacetime(Lux_job *ego, double time_double)
{

	struct param *p = &EGO->param;

	const char *file_name = p->dyst_file;
	const char *time_format = p->time_format;

	/* Dimension names, useful for loops */
	const char dimension_names[4][2] = {"t", "x", "y", "z"};

	/* OpenCL Image properties */
	cl_image_format imgfmt;
	cl_image_desc imgdesc;
	cl_int err;

	/* HDF5 identifiers */
	hid_t file_id;
	hid_t group_id;
	herr_t status;

	clock_t start, end;
	double cpu_time_used;

	start = clock();

	/* The group name is a char, not a double */
	char time[64];

	snprintf(time, sizeof(time), time_format, time_double);

	/* Array of pointers for the coordinates */
	/* There are only 3: x, y, z */
	void *coordinates[3];
	size_t num_points_in_dimensions[3];

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

	/* First, we read the coordinates */
    /* dimension_names[i + 1] because we ignore the time, which is the zeroth */
	for (size_t i = 0; i < 3; i++){
        num_points_in_dimensions[i] = read_variable_from_h5_file_and_return_num_points(
			group_id, dimension_names[i + 1], &coordinates[i]);
		/* This is an error, something didn't work as expected */
		/* We have already printed what */
		if (num_points_in_dimensions[i] <= 0)
			return num_points_in_dimensions[i];
	}

	lux_debug("Read coordiantes\n");

    /* Fill bounding box */
	for (int i = 0; i < 3; i++){
		/* xmin */
		EGO->bounding_box.s[i] = ((cl_float *)coordinates[i])[0];
		/* xmax */
		EGO->bounding_box.s[i + 4] = ((cl_float *)coordinates[i])[num_points_in_dimensions[i] - 1];
	}

	/* Now, we read the Gammas */
	void *Gamma[40];
	size_t expected_num_points = num_points_in_dimensions[0] *
		num_points_in_dimensions[1] *
		num_points_in_dimensions[2];
	size_t num_points;
	size_t index = 0;

	for (size_t i = 0; i < 4; i++)
		for (size_t j = 0; j < 4; j++)
			for (size_t k = j; k < 4; k++) {
				char var_name[256];
				snprintf(var_name, sizeof(var_name), "Gamma_%s%s%s", dimension_names[i],
						 dimension_names[j], dimension_names[k]);

				/* We treat our 3D data as 1D */
				num_points = read_variable_from_h5_file_and_return_num_points(
					group_id, var_name, &Gamma[index]);

				/* This is an error, something didn't work as expected */
				/* We have already printed what */
				if (num_points_in_dimensions[i] <= 0)
					return num_points_in_dimensions[i];

				if (num_points != expected_num_points) {
					lux_print("Number of points in Gammas incosistent with coordinates\n");
					return -1;
				}

				index++;
			}

	lux_debug("Read Gammas\n");

	/* Finally, we create the images */

	imgfmt.image_channel_order = CL_R;         /* use one channel */
	imgfmt.image_channel_data_type = CL_FLOAT; /* each channel is a float */
	imgdesc.image_width = num_points_in_dimensions[0];  /* x */
	imgdesc.image_height = num_points_in_dimensions[1]; /* y */
	imgdesc.image_depth = num_points_in_dimensions[2];  /* z */
	imgdesc.image_row_pitch = 0;
	imgdesc.image_slice_pitch = 0;
	imgdesc.num_mip_levels = 0;
	imgdesc.num_samples = 0;
	imgdesc.mem_object = NULL;
	imgdesc.image_type = CL_MEM_OBJECT_IMAGE3D;

	index = 0;
	for (size_t i = 0; i < 4; i++)
		for (size_t j = 0; j < 4; j++)
			for (size_t k = j; k < 4; k++) {
				EGO->spacetime[index] = clCreateImage(
					EGO->ocl->ctx, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, &imgfmt,
					&imgdesc, Gamma[index], &err);
				/* https://streamhpc.com/blog/2013-04-28/opencl-error-codes/ */
				if (err != CL_SUCCESS) {
					lux_print("Error in creating images\n");
					return err;
				}
				index++;
			}

	lux_debug("Images created\n");

	for (size_t i = 0; i < 3; i++)
		free(coordinates[i]);

	for (size_t i = 0; i < 40; i++)
		free(Gamma[i]);

	status = H5Fclose(file_id);
	if (status != 0) {
		printf("Error in closing HDF5 file");
		return status;
	}

	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

	lux_print("Reading file and creating images took %.5f s\n", cpu_time_used);

	return 0;
}
