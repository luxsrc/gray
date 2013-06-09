// Copyright (C) 2012,2013 Chi-kwan Chan
// Copyright (C) 2012,2013 Steward Observatory
//
// This file is part of GRay.
//
// GRay is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// GRay is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
// or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
// License for more details.
//
// You should have received a copy of the GNU General Public License
// along with GRay.  If not, see <http://www.gnu.org/licenses/>.

#ifndef PARA_H
#define PARA_H

#define WIDTH 1024
#define HARM  1

#ifndef __CUDACC__ // for src/ctrl.cc, src/main.cc, and src/vis.cc
#  define DT_DUMP (-1)
#else // for src/init.cu and src/evolve.cu

// Parameters for geodesic integration
static __constant__ real r_obs     = 30;       // observer radius in GM/c^2
static __constant__ real i_obs     = 30;       // observer theta in degrees
static __constant__ real a_spin    = 0.999;    // dimensionless spin j/mc
static __constant__ real dt_scale  = 1.0 / 32; // typical step size
static __constant__ real epsilon   = 1e-3;     // stop photon
static __constant__ real tolerance = 1e-1;     // if xi+1 > tolerance, fall
                                               // back to forward Euler
// Parameters for radiative transfer
static __constant__ Coord *coord   = NULL;
static __constant__ Field *field   = NULL;

static __constant__ real   R_torus = 6;
static __constant__ real   Omega   = 0.068;

static __constant__ real   Gamma   = 4.0 / 3.0;
static __constant__ real   Tp_Te   = 3;
static __constant__ real   ne_rho  = 1e6;
static __constant__ real   m_BH    = 4.3e6;  /* in unit of solar mass */
static __constant__ real   nu0     = 3.6e11; /* Hz, infrared */

static inline bool config(const char c, const real v)
{
  cudaError_t err = cudaErrorInvalidSymbol;

  switch(c) {
  case 'r': err = cudaMemcpyToSymbol(r_obs,     &v, sizeof(real)); break;
  case 'i': err = cudaMemcpyToSymbol(i_obs,     &v, sizeof(real)); break;
  case 'a': err = cudaMemcpyToSymbol(a_spin,    &v, sizeof(real));
#ifndef DISABLE_GL
    global::a_spin = v;
#endif
    break;
  case 's': err = cudaMemcpyToSymbol(dt_scale,  &v, sizeof(real)); break;
  case 'e': err = cudaMemcpyToSymbol(epsilon,   &v, sizeof(real)); break;
  case 't': err = cudaMemcpyToSymbol(tolerance, &v, sizeof(real)); break;
  case 'G': err = cudaMemcpyToSymbol(Gamma,     &v, sizeof(real)); break;
  case 'n': err = cudaMemcpyToSymbol(nu0,       &v, sizeof(real)); break;
  case 'R': err = cudaMemcpyToSymbol(Tp_Te,     &v, sizeof(real)); break;
  case 'd': err = cudaMemcpyToSymbol(ne_rho,    &v, sizeof(real)); break;
  }

  return cudaSuccess == err;
}
#endif

#endif // PARA_H
