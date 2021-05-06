/*
 * Copyright (C) 2020-2021 Gabriele Bozzola
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
#include <stdlib.h>
#include <hdf5.h>
#include <time.h>


/* Macro that we use to assign horizon properties from attributes */
#define assign_horizon(index)					                                \
/* The indexing for horizons start from 1 */									\
snprintf(attr_name, sizeof(attr_name), "ah_%d", index);					    	\
																				\
/* If the attribute does not exist, then we have to invalidate the				\
 * horizon */																	\
																				\
/* "." means: the attribute is attached to group_id */							\
if (H5Aexists_by_name(group_id, ".", attr_name, H5P_DEFAULT)){					\
																				\
	attr = H5Aopen_by_name(group_id, ".", attr_name, H5P_DEFAULT, H5P_DEFAULT);	\
	if (attr == -1) {															\
		lux_print("Error in opening attribute: %s\n", attr_name);				\
		return attr;															\
	}																			\
																				\
	/* Get size */																\
	atype = H5Aget_type(attr);													\
	if (atype == -1) {															\
		lux_print("Error in reading type of attribute: %s\n", attr_name);		\
		return atype;															\
	}																			\
																				\
	size_t sz_tmp = H5Tget_size(atype);											\
																				\
	/* Check that size is consistent */											\
	if (sz == 0)																\
		sz = sz_tmp;															\
	else																		\
		if (sz != sz_tmp){														\
			lux_print("Precision inconsistent in horizon information.");		\
			return -1;															\
		}																		\
																				\
	aspace = H5Aget_space(attr);												\
	if (aspace == -1) {															\
		lux_print("Error in reading dataspace in attribute: %s\n", attr_name);	\
		return aspace;															\
	}																			\
	size_t npoints = H5Sget_simple_extent_npoints(aspace);						\
																				\
	if (npoints != expected_attr_num){											\
		lux_print("Unexpected number of points in attribute %s\n.");			\
		return -1;																\
	}																			\
																				\
	cl_float* attr_array = (cl_float *)malloc(sz * (int)npoints);				\
	H5Aread(attr, atype, attr_array);											\
	/* The 4th attribute is the maximum radius, if it is not positive,			\
	 * the horizon is not valid */												\
    if (attr_array[4] <= 0){													\
    	EGO->ah_valid_##index[i] = -1;											\
    }else{																		\
    	EGO->ah_valid_##index[i] = 1;											\
    	char *rem;																\
    	cl_float time = strtod(EGO->available_times[i], &rem);					\
    	EGO->ah_centr_##index[i].s0 = time;										\
    	EGO->ah_centr_##index[i].s1 = attr_array[0];							\
    	EGO->ah_centr_##index[i].s2 = attr_array[1];							\
    	EGO->ah_centr_##index[i].s3 = attr_array[2];							\
    	EGO->ah_min_r_##index[i] = attr_array[3];							    \
    	EGO->ah_max_r_##index[i] = attr_array[4];							    \
    }																			\
	H5Aclose(attr);																\
}else{				/* Attribute does not exist */								\
	lux_print("Attribute %s does not exist\n", attr_name);						\
	EGO->ah_valid_##index[i] = -1;								                \
}

/* Compare function, needed for qsort (needed to sort the times in the HDF5 file) */
int compare (const void *a, const void *b)
{
	/* With the help of https://stackoverflow.com/a/3886497 */
	char *rem;
	real a_num = strtod((char*)a, &rem);
	real b_num = strtod((char*)b, &rem);
	return (a_num > b_num) - (a_num < b_num);
}

/** \todo Implement load() */

void
dump(Lux_job *ego, size_t i)
{
	Lux_opencl *ocl = EGO->ocl;

	struct param *p = &EGO->param;
	struct setup *s = &EGO->setup;

	const  size_t sz     = s->precision;
	const  size_t n_data = EGO->n_coor + EGO->n_freq * 2;
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
	/* var_array is a pointer to a pointer to the area of memory
	 * where the data will be written */
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
populate_ego_available_times(Lux_job *ego) {

	/* HDF5 identifiers */
	hid_t file_id;
	hid_t group_id;
	herr_t status;

	hsize_t nobj;

	struct param *p = &EGO->param;
	const char *file_name = p->dyst_file;

	file_id = H5Fopen(file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file_id == -1) {
		lux_print("ERROR: File %s is not a valid HDF5 file\n", file_name);
		return file_id;
	}

	/* Open group corresponding to root */
	group_id = H5Gopen(file_id, "/", H5P_DEFAULT);
	if (group_id == -1) {
		lux_print("ERROR: Could not open root of HDF5 file\n");
		return group_id;
	}

	/* Get all the members of the groups, one at a time */
	status = H5Gget_num_objs(group_id, &nobj);
	if (status != 0) {
		lux_print("ERROR: Could not obtain number of groups in HDF5 file\n");
		return status;
	}

	if (nobj < 2) {
		lux_print("ERROR: Not enough groups in the HDF5 file\n");
		return -1;
	}

	lux_debug("Available times: \n");

	char time_name[MAX_TIME_NAME_LENGTH];

	for (size_t i = 0; i < nobj; i++) {
		H5Gget_objname_by_idx(group_id, i, time_name, MAX_TIME_NAME_LENGTH);
		/* We esclude the "grid" group, which contains the coordinates */
		if (time_name[0] != 'g'){
			char *time_name_in_ego = EGO->available_times[i];
			snprintf(time_name_in_ego, sizeof(time_name), "%s", time_name);
			lux_debug("%s\n", time_name_in_ego);
		}
	}

	/* Now we sort the available_times array in ascending order */
	qsort(EGO->available_times, nobj - 1, sizeof(EGO->available_times[0]), compare);

	lux_debug("Sorted available times: \n");

	for (size_t i = 0; i < nobj - 1; i++){
		lux_debug("%s\n", EGO->available_times[i]);
	}

	char *rem;
	/* Here it is -2 because we have a 'grid' group around, and it has to be after
	 * the numbers (nobj - 2 is the last element of the array) */
	EGO->max_available_time = strtod(EGO->available_times[nobj - 2], &rem);

	return 0;
}

size_t
load_horizons(Lux_job *ego){
	/* Here we load the horizon information from the attributes of the
	 * datasets. */

	struct param *p = &EGO->param;

	const char *file_name = p->dyst_file;

	/* HDF5 identifiers */
	hid_t file_id;
	hid_t group_id, root_group_id;
	hid_t attr;
	hid_t atype;
	hid_t aspace;
	herr_t status;

	hsize_t nobj;

	size_t sz = 0;
	size_t expected_attr_num = 5;

	file_id = H5Fopen(file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file_id == -1) {
		lux_print("ERROR: File %s is not a valid HDF5 file\n", file_name);
		return file_id;
	}

	root_group_id = H5Gopen(file_id, "/", H5P_DEFAULT);
	if (root_group_id == -1) {
		lux_print("ERROR: root group not found!\n");
		return root_group_id;
	}

    /* Get the number of times */
	status = H5Gget_num_objs(root_group_id, &nobj);
	if (status == -1) {
		lux_print("ERROR: reading number of objects!\n");
		return status;
	}

	/* Now we loop over all the times. We have to remove 1 from nobj because we
	 * also have the 'grid' group. */
	for (size_t i = 0; i < nobj - 1; i++) {
		group_id = H5Gopen(root_group_id, EGO->available_times[i], H5P_DEFAULT);
		if (group_id == -1) {
			lux_print("ERROR: problem with group %s\n", EGO->available_times[i]);
			return group_id;
		}
		size_t	num_attr = H5Aget_num_attrs(group_id);
		if (num_attr > MAX_NUM_HORIZONS){
			lux_print("ERROR: too many horizons! (found %d attributes) \n", num_attr);
			return -1;
		}
		char attr_name[256];

		assign_horizon(1);
		assign_horizon(2);
		assign_horizon(3);

		status = H5Gclose(group_id);
		if (status != 0) {
			lux_print("Error in closing group: %s", EGO->available_times[i]);
			return status;
		}
	}

	return 0;

}


size_t
load_coordinates(Lux_job *ego){
	/* Here we load the coordinates from the 'grid' dataset in the HDF5 file */

	/* OpenCL Image properties */
	cl_image_format imgfmt;
	cl_image_desc imgdesc;
	cl_int err;

	struct param *p = &EGO->param;

	const char *file_name = p->dyst_file;

	/* Dimension names, useful for loops */
	const char dimension_names[4][2] = {"t", "x", "y", "z"};

	/* HDF5 identifiers */
	hid_t file_id;
	hid_t group_id;
	herr_t status;

	/* Array of pointers for the coordinates */
	/* There are only 3: x, y, z */
	void *coordinates[3];

	file_id = H5Fopen(file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file_id == -1) {
		lux_print("ERROR: File %s is not a valid HDF5 file\n", file_name);
		return file_id;
	}

	group_id = H5Gopen(file_id, "grid", H5P_DEFAULT);
	if (group_id == -1) {
		lux_print("ERROR: grid group not found!\n");
		return group_id;
	}

	/* First, we read the coordinates */
	/* dimension_names[i + 1] because we ignore the time, which is the zeroth */
	for (size_t i = 0; i < 3; i++){
		EGO->num_points.s[i + 1] = read_variable_from_h5_file_and_return_num_points(
			group_id, dimension_names[i + 1], &coordinates[i]);
		/* This is an error, something didn't work as expected */
		/* We have already printed what */
		if (EGO->num_points.s[i + 1] <= 0)
			return EGO->num_points.s[i + 1];
	}

	lux_debug("Read coordiantes\n");

	/* Fill spatial bounding box */
	for (int i = 1; i < 4; i++){
		/* xmin */
		EGO->bounding_box.s[i] = ((cl_float *)coordinates[i - 1])[0];
		/* xmax */
		EGO->bounding_box.s[i + 4] = ((cl_float *)coordinates[i - 1])[EGO->num_points.s[i] - 1];
	}

	return 0;
}

void
copy_snapshot(Lux_job *ego, size_t to_t1){
	/* This function copies over the snapshot in _t2 to _t1 if to_t1 is true,
	 * otherwise from _t1 to _t2.  Before doing this, the memory is released.
	 * We assume that data is already defined before copying. */
	size_t index = 0;
	for (size_t i = 0; i < 4; i++)
		for (size_t j = 0; j < 4; j++)
			for (size_t k = j; k < 4; k++) {
				if (to_t1){
					/* TODO: Error checking on clReleaseMemObject */
					clReleaseMemObject(EGO->spacetime_t1[index]);
					EGO->spacetime_t1[index] = EGO->spacetime_t2[index];
				}else{
					/* TODO: Error checking on clReleaseMemObject */
					clReleaseMemObject(EGO->spacetime_t2[index]);
					EGO->spacetime_t2[index] = EGO->spacetime_t1[index];
				}
				index++;
			}

}

size_t
load_snapshot(Lux_job *ego, size_t time_snapshot_index, size_t load_in_t1){

	/* If load_in_t1 is true, then fill the t1 slot, otherwise, fill the t2 slot. */

	/* TODO: Add support to compressed HDF5 files */
	/* https://support.hdfgroup.org/ftp/HDF5/examples/examples-by-api/hdf5-examples/1_10/C/H5D/h5ex_d_shuffle.cgo */

	struct param *p = &EGO->param;

	const char *file_name = p->dyst_file;

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

	char *time = EGO->available_times[time_snapshot_index];

	start = clock();

	file_id = H5Fopen(file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file_id == -1) {
		lux_print("ERROR: File %s is not a valid HDF5 file\n", file_name);
		return file_id;
	}
	/* Open group corresponding to time */
	group_id = H5Gopen(file_id, time, H5P_DEFAULT);
	if (group_id == -1) {
		lux_print("ERROR: Time %s not found\n", time);
		return group_id;
	}

	lux_debug("Reading time %s\n", time);

	/* Now, we read the Gammas */
	void *Gamma[40];
	size_t num_points;
	size_t index = 0;
	const size_t expected_num_points = EGO->num_points.s[1] *
		EGO->num_points.s[2] *
		EGO->num_points.s[3];

	for (size_t i = 0; i < 4; i++)
		for (size_t j = 0; j < 4; j++)
			for (size_t k = j; k < 4; k++) {
				char var_name[256];
				snprintf(var_name, sizeof(var_name), "Gamma_%s%s%s", dimension_names[i],
						 dimension_names[j], dimension_names[k]);

				/* We treat our 3D data as 1D */
				num_points = read_variable_from_h5_file_and_return_num_points(
					group_id, var_name, &Gamma[index]);

				/* 	for (size_t kk=0; kk < num_points; kk++){ */
				/* 		float val = ((float*)Gamma[index])[kk]; */
				/* 		if (val > 1) printf("Gamma kk %d, %.16g\n", kk, val); */
				/* } */

				/* This is an error, something didn't work as expected */
				/* We have already printed what */
				if (num_points <= 0)
					return num_points;

				if (num_points != expected_num_points) {
					lux_print("Number of points in Gammas inconsistent with coordinates\n");
					return -1;
				}
				index++;
			}

	lux_debug("Read Gammas\n");

	/* Read metric */
	void *g[10];

	index = 0;
	for(size_t i = 0; i < 4; i++)
		for(size_t j = i; j < 4; j++) {
			char var_name[256];
			snprintf(var_name, sizeof(var_name), "g_%s%s", dimension_names[i], dimension_names[j]);

			/* We treat our 3D data as 1D */
			num_points = read_variable_from_h5_file_and_return_num_points(
				group_id, var_name, &g[index]);

			/* This is an error, something didn't work as expected */
			/* We have already printed what */
			if (num_points <= 0)
				return num_points;

			if (num_points != expected_num_points) {
				lux_print("Number of points in the metric inconsistent with coordinates\n");
				return -1;
			}
			index++;
		}

	lux_debug("Read metric\n");

	void *rho;
	{
		/* We treat our 3D data as 1D */
		num_points = read_variable_from_h5_file_and_return_num_points(
			group_id, "rho", &rho);

		/* This is an error, something didn't work as expected */
		/* We have already printed what */
		if (num_points <= 0)
			return num_points;

		if (num_points != expected_num_points) {
			lux_print("Number of points in rho inconsistent with coordinates\n");
			return -1;
		}
	}

	/* Read magnetic field */
	void *b[4];

	for(size_t j = 0; j < 4; j++) {
		char var_name[256];
		snprintf(var_name, sizeof(var_name), "b_%s", dimension_names[j]);

		/* We treat our 3D data as 1D */
		num_points = read_variable_from_h5_file_and_return_num_points(
			group_id, var_name, &b[j]);

		/* This is an error, something didn't work as expected */
		/* We have already printed what */
		if (num_points <= 0)
			return num_points;

		if (num_points != expected_num_points) {
			lux_print("Number of points in the magnetic field inconsistent with coordinates\n");
			return -1;
		}
	}

	lux_debug("Read fluid\n");

	/* Finally, we create the images */

	imgfmt.image_channel_order = CL_R;         /* use one channel */
	imgfmt.image_channel_data_type = CL_FLOAT; /* each channel is a float */
	imgdesc.image_type = CL_MEM_OBJECT_IMAGE3D;
	imgdesc.image_width = EGO->num_points.s[1];  /* x */
	imgdesc.image_height = EGO->num_points.s[2]; /* y */
	imgdesc.image_depth = EGO->num_points.s[3];  /* z */
	imgdesc.image_row_pitch = 0;
	imgdesc.image_slice_pitch = 0;
	imgdesc.num_mip_levels = 0;
	imgdesc.num_samples = 0;
	imgdesc.buffer = NULL;

	index = 0;
	for (size_t i = 0; i < 4; i++)
		for (size_t j = 0; j < 4; j++)
			for (size_t k = j; k < 4; k++) {
				/* We fill _t1 only the first time, when snapshot_index = 0,
				 * in all the other cases we fill _t2, and then we shift the pointers.*/
				if (load_in_t1){
					EGO->spacetime_t1[index] = clCreateImage(
						EGO->ocl->ctx, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, &imgfmt,
						&imgdesc, Gamma[index], &err);
				}else{
					EGO->spacetime_t2[index] = clCreateImage(
						EGO->ocl->ctx, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, &imgfmt,
						&imgdesc, Gamma[index], &err);
				}
				/* https://streamhpc.com/blog/2013-04-28/opencl-error-codes/ */
				if (err != CL_SUCCESS) {
					lux_print("Error in creating images\n");
					return err;
				}
				index++;
			}

	for (size_t i = 0; i < 4; i++)
		for (size_t j = i; j < 4; j++) {
			/* We fill _t1 only the first time, when snapshot_index = 0,
			 * in all the other cases we fill _t2, and then we shift the pointers.*/
			if (load_in_t1){
				EGO->spacetime_t1[index] = clCreateImage(
					EGO->ocl->ctx, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, &imgfmt,
					&imgdesc, g[index-40], &err);
			}else{
				EGO->spacetime_t2[index] = clCreateImage(
					EGO->ocl->ctx, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, &imgfmt,
					&imgdesc, g[index-40], &err);
			}
			/* https://streamhpc.com/blog/2013-04-28/opencl-error-codes/ */
			if (err != CL_SUCCESS) {
				lux_print("Error in creating images\n");
				return err;
			}
			index++;
		}

	{
		/* We fill _t1 only the first time, when snapshot_index = 0,
		 * in all the other cases we fill _t2, and then we shift the pointers.*/
		if (load_in_t1){
			EGO->spacetime_t1[index] = clCreateImage(
				EGO->ocl->ctx, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, &imgfmt,
				&imgdesc, rho, &err);
		}else{
			EGO->spacetime_t2[index] = clCreateImage(
				EGO->ocl->ctx, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, &imgfmt,
				&imgdesc, rho, &err);
		}
		/* https://streamhpc.com/blog/2013-04-28/opencl-error-codes/ */
		if (err != CL_SUCCESS) {
			lux_print("Error in creating images\n");
			return err;
		}
	}
	index++;

	for (size_t j = 0; j < 4; j++) {
		/* We fill _t1 only the first time, when snapshot_index = 0,
		 * in all the other cases we fill _t2, and then we shift the pointers.*/
		if (load_in_t1){
			EGO->spacetime_t1[index] = clCreateImage(
				EGO->ocl->ctx, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, &imgfmt,
				&imgdesc, g[j], &err);
		}else{
			EGO->spacetime_t2[index] = clCreateImage(
				EGO->ocl->ctx, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, &imgfmt,
				&imgdesc, g[j], &err);
		}
		/* https://streamhpc.com/blog/2013-04-28/opencl-error-codes/ */
		if (err != CL_SUCCESS) {
			lux_print("Error in creating images\n");
			return err;
		}
		index++;
	}

	lux_debug("Images created\n");

	for (size_t i = 0; i < 40; i++)
		free(Gamma[i]);

	for (size_t i = 0; i < 10; i++)
		free(g[i]);

	free(rho);

	for (size_t i = 0; i < 4; i++)
		free(b[i]);

	status = H5Fclose(file_id);
	if (status != 0) {
		lux_print("Error in closing HDF5 file");
		return status;
	}

	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

	lux_print("Reading file and creating images for time %s took %.5f s\n",
	          time,
	          cpu_time_used);

	return 0;
}
