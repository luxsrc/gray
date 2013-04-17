// Copyright (C) 2013 Chi-kwan Chan
// Copyright (C) 2013 Steward Observatory
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

#ifndef HARM_H
#define HARM_H

#include <cstdio>

typedef struct {
  // size_t i, j, k;
  // double x1, x2, x3;
  // double v1, v2, v3;
  // double gcon[4][4];
  // double gcov[4][4];
  // double gdet;
  // double ck[4];
  // double dxdxp[4][4];
  float gcov [4][4];
  float dxdxp[4][4];
} Coord;

typedef struct {
  float rho;
  float u;
  float jet1; // = -u_t
  float jet2; // = -T^r_t / (rho u^r)
  float ucont;
  float v1, v2, v3;
  float B1, B2, B3;
} Field;

namespace harm {
  extern double t,  R0;
  extern size_t n1, n2, n3;
  extern Coord *coord;
  extern Field *field;
}

extern Coord *load_coord(const char *);
extern Field *load_field(const char *);

#endif // HARM_H
