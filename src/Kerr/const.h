// Copyright (C) 2012--2014 Chi-kwan Chan
// Copyright (C) 2012--2014 Steward Observatory
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

#ifndef CONST_H
#define CONST_H

#include "harm.h"

#define DT_DUMP (-100)
#define N_R     264 // large enough to hold our grids
//efine N_THETA 128 // large enough to hold our grids
//efine N_IN    64  // inner theta-grid is stored in constant memory
#define N_NU    5

typedef struct {
  // Parameters for geodesic integration
  real imgsz;     // image size in GM/c^2
  real r_obs;     // observer radius in GM/c^2
  real i_obs;     // observer theta in degrees
  real a_spin;    // dimensionless spin j/mc
  real dt_scale;  // typical step size
  real epsilon;   // stop photon
  real tolerance; // if xi+1 > tolerance, fall back to forward Euler

  // Parameters for radiative transfer
  real   m_BH, ne_rho, threshold, tgas_max, Ti_Te_d, Ti_Te_f;
  real   nu0[N_NU];
  size_t n_nu;

  Coord *coord;
  Field *field;
  size_t n_rx, n_r, n_theta, n_phi;
  real   Gamma;
  real   r[N_R];
#if defined(N_IN) && defined(N_THETA)
  real   theta[N_IN * N_THETA];
#endif
} Const;

namespace harm {
  extern Coord *load_coord(Const &, const char *);
  extern Field *load_field(Const &, const char *);
  extern bool   load      (Const &, const char *);
  extern bool   setx      (Const &, const char *);
  extern bool   using_harm;
  extern size_t n_nu;
}

#endif // CONST_H
