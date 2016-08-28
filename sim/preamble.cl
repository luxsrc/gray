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
 ** Preamble: useful OpenCL macros and functions
 **
 ** GRay2 uses OpenCL's just-in-time compilation feature to implement
 ** run-time configurable algorithms.  In this preamble we provide
 ** OpenCL macros and functions that help implementing the other parts
 ** of GRay2.
 **/

#define EPSILON 1e-32

/** Helper macros to write equations for vector of length n_vars **/
#define EACH(s) for(whole _e_ = 0; _e_ < n_chunk; ++_e_) E(s)
#define E(s) ((realE *)&(s))[_e_]

/** Turn an expression into a local variable that can be passed to function **/
#define X(x) ({ struct state _; EACH(_) = (x); _; })
