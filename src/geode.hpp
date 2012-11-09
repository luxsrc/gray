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

#ifndef GEODE_HPP
#define GEODE_HPP

#include <iostream>

#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glut.h>
#endif

#if defined(DOUBLE) || defined(OUBLE) /* So -DOUBLE works */
  typedef double Real;
#else
  typedef float Real;
#endif

typedef struct {
  Real x, y, z;
  Real u, v, w;
} State;

namespace global {
  extern size_t n;
  extern State *s;
}

State *init(size_t);
int setup(int &, char **);
int solve(void);
void vis(void);

#endif
