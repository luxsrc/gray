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
#ifndef _GRAY_INITCOND_H_
#define _GRAY_INITCOND_H_

#include <lux.h>
#include <lux/opencl.h>

typedef struct LuxSgray_initcond Lux_gray_initcond;
typedef struct LuxOgray_initcond Lux_gray_initcond_opts;

struct LuxSgray_initcond {
	size_t (*getn)(Lux_gray_initcond *);
	cl_mem (*init)(Lux_gray_initcond *);
};

struct LuxOgray_initcond {
	size_t            nque;
	cl_command_queue *que;
	void             *opts;
};

#endif /* _GRAY_INITCOND_H */
