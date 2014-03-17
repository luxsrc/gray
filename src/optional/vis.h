// Copyright (C) 2014 Chi-kwan Chan
// Copyright (C) 2014 Steward Observatory
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

#ifndef VIS_H
#define VIS_H // Translate macros that are passed from the Makefile

class Para;

#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glut.h>
#endif
#include <GLFW/glfw3.h>

namespace vis {
  extern GLFWwindow *window;
  extern int    width, height;
  extern float  ratio, ax, ly, az, a_spin;
  extern double dt_saved;
  extern int    shader;
  extern bool   draw_body;
  extern Para  *p;

  extern void mktexture(GLuint[]);
  extern void mkshaders(GLuint[]);

  extern void resize  (GLFWwindow *, int, int);
  extern void keyboard(GLFWwindow *, int, int, int, int);
  extern void mouse   (GLFWwindow *, double, double);

#ifdef ENABLE_PRIME
  extern void sense();
  extern void track();
#endif
#ifdef ENABLE_LEAP
  extern void sense();
#endif
}

#endif // VIS_H
