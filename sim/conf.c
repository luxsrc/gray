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

/** \page newopts New Run-Time Options
 **
 ** Turn hard-wired constants into run-time options
 **
 ** GRay2 uses the lux framework and hence follows lux's approach to
 ** support many run-time options.  To turn hard-wired constants into
 ** run-time options, one needs to
 **
 **   -# Add an option table ".otab" file
 **   -# Add the automatically generated structure to "sim/gray.h"
 **   -# Add the automatically generated configure function to "sim/conf.c"
 **/
#include "gray.h"

int
conf(Lux_job *ego, const char *restrict arg)
{
	lux_debug("GRay2: configuring job %p with argument \"%s\"\n", ego, arg);

	return icond_config(&EGO->icond, arg) &&
	       param_config(&EGO->param, arg) &&
	       setup_config(&EGO->setup, arg);
}
