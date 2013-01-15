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

#include <shaders.h>

#define GL_VERTEX_PROGRAM_POINT_SIZE_NV 0x8642
#define STR1NG(x) #x
#define STRING(x) STR1NG(x)

static GLuint vbo = 0; // OpenGL Vertex Buffer Object
static GLuint shader[2];

static float ax = 270, az = 90;
static float ly =-50;

static double dt_stored = 0.0;
static int last_x = 0, last_y = 0;
static int left   = 0, right  = 0;

static unsigned which = 1;

static GLuint compile(const char *src, GLenum type)
{
  GLuint s = glCreateShader(type);
  glShaderSource(s, 1, &src, 0);
  glCompileShader(s);
  return s;
}

static unsigned char *mkimg(int n)
{
  unsigned char *img = new unsigned char[4 * n * n];

  for(int h = 0, i = 0; i < n; ++i) {
    double x  = 2 * (i + 0.5) / n - 1;
    double x2 = x * x;
    for(int j = 0; j < n; ++j, h += 4) {
      double y  = 2 * (j + 0.5) / n - 1;
      double r2 = x2 + y * y;
      if(r2 > 1) r2 = 1;
      img[h] = img[h+1] = img[h+2] = img[h+3] =
        255 * ((2 * sqrt(r2) - 3) * r2 + 1);
    }
  }

  return img;
}

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
  glDrawArrays(GL_POINTS, 0, global::n);
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
  using namespace global;

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

void vis(GLuint vbo_in)
{
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);

  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);

  unsigned int texture;
  glGenTextures(1, (GLuint *)&texture);
  glBindTexture(GL_TEXTURE_2D, texture);

  glTexParameteri(GL_TEXTURE_2D,
                  GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
  glTexParameteri(GL_TEXTURE_2D,
                  GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D,
                  GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  unsigned char *img = mkimg(64);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 64, 64, 0,
               GL_RGBA, GL_UNSIGNED_BYTE, img);

  glActiveTextureARB(GL_TEXTURE0_ARB);
  glBindTexture(GL_TEXTURE_2D, texture);

  glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE);

  shader[0] = glCreateProgram();
  glAttachShader(shader[0], compile(STRING(VERTEX_SHADER), GL_VERTEX_SHADER));
  glLinkProgram(shader[0]);

  shader[1] = glCreateProgram();
  glAttachShader(shader[1], compile(STRING(VERTEX_SHADER), GL_VERTEX_SHADER));
  glAttachShader(shader[1], compile(STRING(PIXEL_SHADER), GL_FRAGMENT_SHADER));
  glLinkProgram(shader[1]);

  glEnable(GL_DEPTH_TEST);
  glClearColor(0.0, 0.0, 0.0, 1.0);

  vbo = vbo_in;
}

#endif // !DISABLE_GL
