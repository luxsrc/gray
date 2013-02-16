// Copyright (C) 2012 Chi-kwan Chan
// Copyright (C) 2012 Steward Observatory
//
// This file is part of geode.
//
// Geode is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Geode is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
// or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
// License for more details.
//
// You should have received a copy of the GNU General Public License
// along with geode.  If not, see <http://www.gnu.org/licenses/>.

#ifndef PARA_H
#define PARA_H

#define N_ALPHA   1024
#define N_BETA    768
#define N_DEFAULT (N_ALPHA * N_BETA)
#define DT_DUMP   (-1)              // default dump time

__device__ __constant__ real r_obs    = 20;        // observer radius in GM/c^2
__device__ __constant__ real i_obs    = 30;        // observer theta in degrees
__device__ __constant__ real a_spin   = 0.999;     // dimensionless spin j/mc
__device__ __constant__ real dt_scale = 1.0 / 256; // typical step size
__device__ __constant__ real delta    = 1e-3;      // stop photon

#endif // PARA_H
