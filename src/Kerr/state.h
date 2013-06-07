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

#ifndef STATE_H
#define STATE_H

#define PARTICLE_TIME t

typedef struct {
  real t, r, theta, phi;
  real kr, ktheta;
  real bimpact;       // impact parameter defined as L / E, constant
  real alpha0, beta0; // initial location in the image plane
  real nu0;           // initial photon frequency
  real I, Q, U, V;    // Stokes parameters
  real tau;           // optical depth
} State;

#endif // STATE_H
