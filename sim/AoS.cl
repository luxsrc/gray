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
 ** Array-of-Structures global index
 **
 ** GRay2 uses OpenCL's just-in-time compilation feature to implement
 ** run-time configurable algorithms.  In this file we provide global
 ** index for Array-of-Structures memory order.
 **/

#define DATA(g, s) data[g * n_data + s]
#define INFO(g, s) info[g * n_info + s]
