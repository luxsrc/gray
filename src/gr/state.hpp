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

#ifndef STATE_HPP
#define STATE_HPP

#define R_SCHW ((Real)2.0)
#define A_SPIN ((Real)0.999)

typedef struct {
  Real t, r, theta, phi;
  Real kr, ktheta;
  Real bimpact; // impact parameter defined as L / E
} State;

#endif // STATE_HPP
