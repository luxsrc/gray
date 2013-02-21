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

#ifndef __CUDACC__ // for src/main.cc
#  define N_DEFAULT (181 * 6000)
#  define DT_DUMP   (-100)
#else // for src/init.cu and src/evolve.cu
static __constant__ real r_obs     = 1000;      // observer radius in GM/c^2
static __constant__ real i_obs     = 30;        // observer theta in degrees
static __constant__ real a_spin    = 0.999;     // dimensionless spin j/mc
static __constant__ real dt_scale  = 1.0 / 256; // typical step size
static __constant__ real epsilon   = 1e-6;      // stop photon
static __constant__ real tolerance = 1e-3;      // if xi+1 > tolerance, fall
                                                // back to forward Euler
static inline bool config(const char c, const real v)
{
  cudaError_t err = cudaErrorInvalidSymbol;

  switch(c) {
  case 'r': err = cudaMemcpyToSymbol(r_obs,     &v, sizeof(real)); break;
  case 'i': err = cudaMemcpyToSymbol(i_obs,     &v, sizeof(real)); break;
  case 'a': err = cudaMemcpyToSymbol(a_spin,    &v, sizeof(real)); break;
  case 's': err = cudaMemcpyToSymbol(dt_scale,  &v, sizeof(real)); break;
  case 'e': err = cudaMemcpyToSymbol(epsilon,   &v, sizeof(real)); break;
  case 't': err = cudaMemcpyToSymbol(tolerance, &v, sizeof(real)); break;
  }

  return cudaSuccess == err;
}
#endif

#endif // PARA_H
