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

#include "../gray.h"

#include <cmath>
#include <para.h>
#include <shader.h> // to get vertex and color pointer offsets

#ifndef WIDTH
#define WIDTH 512
#endif

#ifndef HEIGHT
#define HEIGHT 512
#endif

#ifndef VERTEX_POINTER_OFFSET
#define VERTEX_POINTER_OFFSET 0
#endif

#ifndef COLOR_POINTER_OFFSET
#define COLOR_POINTER_OFFSET  3
#endif

#define GL_VERTEX_PROGRAM_POINT_SIZE_NV 0x8642

extern void mktexture(GLuint[]);
extern void mkshaders(GLuint[]);
extern int  getctrl();
extern void regctrl();

namespace global {
  float ratio = 1, a_spin = 0.999;
}

static size_t  n;
static GLuint  vbo; // OpenGL Vertex Buffer Object
static GLuint  shader[2], texture;
static GLfloat width, height;

static void display(void)
{
  glViewport(0, 0, width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(27.0, global::ratio, 1.0, 1.0e6);
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

  // DONE
  glUseProgram(0);
  if(GL_NO_ERROR != glGetError())
    error("callback: display(): fail to visualize simulation\n");
  glutSwapBuffers();
}

static void reshape(int w, int h)
{
  global::ratio = (width = w) / (height = h);
}

void setup(int &argc, char *argv[])
{
  glutInit(&argc, argv);
  glutInitWindowSize(WIDTH, HEIGHT);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutCreateWindow(argv[0]);
  if(GL_NO_ERROR != glGetError())
    error("main(): fail to initialize OpenGL/GLUT\n");
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
