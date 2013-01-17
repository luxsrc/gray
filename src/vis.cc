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

#include "geode.h"

#ifndef DISABLE_GL

#include <cstdlib>
#include <cmath>
#include <shader.h>

#ifndef VERTEX_POINTER_OFFSET
#define VERTEX_POINTER_OFFSET 0
#endif

#ifndef COLOR_POINTER_OFFSET
#define COLOR_POINTER_OFFSET  3
#endif

#define GL_VERTEX_PROGRAM_POINT_SIZE_NV 0x8642

static size_t n = 0;
static GLuint vbo = 0; // OpenGL Vertex Buffer Object

static GLuint shader[2];
static GLuint texture;

static float ax = 270, az = 90;
static float ly =-50;

static int last_x = 0, last_y = 0;
static int left   = 0, right  = 0;

static unsigned which = 1;

static void display(void)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();

  glRotatef(-90, 1, 0, 0);
  glTranslatef(0, -ly, 0);
  glRotatef(-(az- 90), 1, 0, 0);
  glRotatef(-(ax-270), 0, 0, 1);

  // Draw wire sphere, i.e., the "black hole"
  glColor3f(0.0, 1.0, 0.0);
#ifdef A_SPIN
  glutWireSphere(1.0 + sqrt(1.0 - A_SPIN * A_SPIN), 32, 16);
#else
  glutWireSphere(1.0, 32, 16);
#endif

  // Draw particles, i.e., photon locations
  glUseProgram(shader[which]);

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

static void keyboard(unsigned char key, int x, int y)
{
  static double dt_stored = 0.0;

  switch(key) {
  case 27 : // ESCAPE key
  case 'q':
  case 'Q':
    exit(0);
    break;
  case 's':
  case 'S':
    if(++which >= sizeof(shader) / sizeof(GLuint)) which = 0;
    break;
  case 'r':
  case 'R':
    if(dt_dump == 0.0)
      dt_stored *= -1; // fall through
    else {
      dt_dump *= -1;
      break;
    }
  case 'p':
  case 'P':
    double temp = dt_stored;
    dt_stored = dt_dump;
    dt_dump = temp;
    break;
  }
}

static void mouse(int b, int s, int x, int y)
{
  last_x = x;
  last_y = y;

  switch(b) {
  case GLUT_LEFT_BUTTON:
    left  = (s == GLUT_DOWN);
    break;
  case GLUT_RIGHT_BUTTON:
    right = (s == GLUT_DOWN);
    break;
  }
}

static void motion(int x, int y)
{
  int dx = x - last_x; last_x = x;
  int dy = y - last_y; last_y = y;

  if(right)
    ly -= 0.05 * dy;
  else if(left) {
    az -= 0.5 * dy;
    ax -= 0.5 * dx;
  }

  glutPostRedisplay();
}

void vis(GLuint vbo_in, size_t n_in)
{
  vbo = vbo_in;
  n   = n_in;

  mktexture(&texture);
  mkshaders(shader);

  glEnable(GL_DEPTH_TEST);
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE);

  glutDisplayFunc(display);
  glutReshapeFunc(reshape);

  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
}

#endif // !DISABLE_GL
