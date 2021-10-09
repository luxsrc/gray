/*
 * Copyright (C) 2021 Gabriele Bozzola and Chi-kwan Chan
 * Copyright (C) 2021 Steward Observatory
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
#ifndef _GRAY_H_
#define _GRAY_H_

#include <lux.h>
#include <lux/job.h>
#include <lux/opencl.h>
#include <lux/task.h>

#include "gray_opts.h"
#include "infcam_opts.h"

struct gray {
	Lux_job super;

	struct gray_opts opts;
	union {
		struct infcam_opts infcam;
	} initcond;

	Lux_opencl *ocl;

	Lux_task *gi;
	Lux_task *flow;
	Lux_task *rt;
};

#endif /* _GRAY_H */
