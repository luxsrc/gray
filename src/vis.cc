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

#include "gray.h"

#ifndef DISABLE_GL
#include <cmath>
#include <shader.h> // to get vertex and color pointer offsets

#ifndef VERTEX_POINTER_OFFSET
#define VERTEX_POINTER_OFFSET 0
#endif

#ifndef COLOR_POINTER_OFFSET
#define COLOR_POINTER_OFFSET  3
#endif

#define GL_VERTEX_PROGRAM_POINT_SIZE_NV 0x8642

static size_t n;
static GLuint vbo; // OpenGL Vertex Buffer Object
static GLuint shader[2];
static GLuint texture;

static void display(void)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  track();
  int i = getctrl();

  // Draw wire sphere, i.e., the "black hole"
  glColor3f(0.0, 1.0, 0.0);
#ifdef A_SPIN
  glutWireSphere(1.0 + sqrt(1.0 - A_SPIN * A_SPIN), 32, 16);
#else
  glutWireSphere(1.0, 32, 16);
#endif

  // Draw particles, i.e., photon locations
  glUseProgram(shader[i]);

  glEnable(GL_POINT_SPRITE_ARB);
  glEnable(GL_BLEND);
  glDepthMask(GL_FALSE);
  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexPointer(3, GL_REAL, sizeof(State),
                  (char *)(VERTEX_POINTER_OFFSET * sizeof(real)));
  glColorPointer (3, GL_REAL, sizeof(State),
                  (char *)(COLOR_POINTER_OFFSET  * sizeof(real)));
  glDrawArrays(GL_POINTS, 0, n);

  glDisableClientState(GL_COLOR_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);
  glDepthMask(GL_TRUE);
  glDisable(GL_BLEND);
  glDisable(GL_POINT_SPRITE_ARB);

  glUseProgram(0);

  if(GL_NO_ERROR != glGetError())
    error("callback: display(): fail to visualize simulation\n");

  glutSwapBuffers();
}

static void reshape(int w, int h)
{
  glViewport(0, 0, w, h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(27.0, (double)w / h, 1.0, 2500.0);
  glMatrixMode(GL_MODELVIEW);
}

void vis(GLuint vbo_in, size_t n_in)
{
  n   = n_in;
  vbo = vbo_in;
  mkshaders(shader);
  mktexture(&texture);

  glEnable(GL_DEPTH_TEST);
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE);

  regctrl();
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);

  if(GL_NO_ERROR != glGetError())
    error("vis(): fail to setup visualization\n");
}

#endif // !DISABLE_GL
