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

#include "geode.hpp"

#ifndef DISABLE_GL

#include <cuda_gl_interop.h> // OpenGL interoperability runtime API

typedef struct {
  float x, y, z;
  float r, g, b;
} Point;

static GLuint vbo = 0; // OpenGL Vertex Buffer Object
static struct cudaGraphicsResource *res = NULL;

static float ax = 270, az = 90;
static float ly =-50;

static double dt_stored = 0.0;
static int last_x = 0, last_y = 0;
static int left   = 0, right  = 0;

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
  glBindBuffer(GL_ARRAY_BUFFER, vbo);

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);
  {
    const size_t full = sizeof(Point);
    const size_t half = full / 2;

    glVertexPointer(3, GL_FLOAT, full, 0);
    glColorPointer (3, GL_FLOAT, full, (char *)half);

    glDrawArrays(GL_POINTS, 0, global::n);
  }
  glDisableClientState(GL_COLOR_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);

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

static void setup(void)
{
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);

  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);

  glEnable(GL_DEPTH_TEST);
  glClearColor(0.0, 0.0, 0.0, 1.0);

  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(Point) * global::n, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0); // unbind all buffer

  cudaGraphicsGLRegisterBuffer(&res, vbo, cudaGraphicsMapFlagsWriteDiscard);
}

#include <map.cu>

static __global__ void kernel(Point *p, const State *s, size_t n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < n) p[i] = map(s[i]);
}

void vis(void)
{
  if(!vbo) setup();

  size_t size = 0;
  void  *head = NULL;

  cudaGraphicsMapResources(1, &res, 0);
  cudaGraphicsResourceGetMappedPointer(&head, &size, res);
  {
    using namespace global;

    const int bsz = 256;
    const int gsz = (n - 1) / bsz + 1;

    kernel<<<gsz, bsz>>>((Point *)head, s, n);
  }
  cudaGraphicsUnmapResources(1, &res, 0); // unmap resource

  cudaDeviceSynchronize();
}

#endif // !DISABLE_GL
