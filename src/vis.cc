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

namespace global {
  float a_spin = 0.999;
}

static size_t  n;
static GLuint  vbo; // OpenGL Vertex Buffer Object
static GLuint  shader[3], texture;
static GLfloat width = 1024, height = 512;

static void display(void)
{
  // LEFT VIEW PORT
  glViewport(0, 0, width / 2, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(27.0, global::ratio, 1.0, 2500.0);
  glMatrixMode(GL_MODELVIEW);
  const int i = getctrl();

  // Draw wire sphere, i.e., the "black hole"
  glColor3f(0.0, 1.0, 0.0);
  glutWireSphere(1.0 + sqrt(1.0 - global::a_spin * global::a_spin), 32, 16);

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

  // RIGHT VIEW PORT
  glViewport(width / 2, 0, width / 2, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // Draw traced image by shader; TODO: we should use glTexSubImage2D()
  glUseProgram(shader[2]);

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexPointer(3, GL_REAL, sizeof(State), (char *)(7  * sizeof(real)));
  glColorPointer (3, GL_REAL, sizeof(State), (char *)(10 * sizeof(real)));
  glDrawArrays(GL_POINTS, 0, n);

  glDisableClientState(GL_COLOR_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);

  // DONE
  glUseProgram(0);
  if(GL_NO_ERROR != glGetError())
    error("callback: display(): fail to visualize simulation\n");
  glutSwapBuffers();
}

static void reshape(int w, int h)
{
  global::ratio = (width = w) / (height = h) / 2;
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
