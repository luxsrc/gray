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

static GLuint vbo = 0; // OpenGL Vertex Buffer Object
static struct cudaGraphicsResource *res = NULL;

static void display(void)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Draw wire sphere, i.e., the "black hole"
  glColor3f(0.0, 1.0, 0.0);
  glutWireSphere(2.0, 32, 16);

  // Draw particles, i.e., photon locations
  glBindBuffer(GL_ARRAY_BUFFER, vbo);

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);
  {
    const size_t full = 6 * sizeof(float);
    const size_t half = 3 * sizeof(float);

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
  gluPerspective(27.0, (double)w / h, 1.0, 100.0);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(0.0,-50.0,  0.0,
            0.0,  0.0,  0.0,
            0.0,  0.0,  1.0);
}

static void keyboard(unsigned char key, int x, int y)
{
  switch(key) {
  case 27 : // ESCAPE key
  case 'q':
  case 'Q':
    exit(0);
    break;
  }
}

static void setup(void)
{
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutKeyboardFunc(keyboard);

  glEnable(GL_DEPTH_TEST);
  glClearColor(0.0, 0.0, 0.0, 1.0);

  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(Point) * global::n, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0); // unbind all buffer

  cudaGraphicsGLRegisterBuffer(&res, vbo, cudaGraphicsMapFlagsWriteDiscard);
}

void vis(void)
{
  if(!vbo) setup();

  size_t size = 0;
  void  *head = NULL;

  cudaGraphicsMapResources(1, &res, 0);
  cudaGraphicsResourceGetMappedPointer(&head, &size, res);
  map((Point *)head, global::s, global::n);
  cudaGraphicsUnmapResources(1, &res, 0); // unmap resource

  cudaDeviceSynchronize();
}

#endif // !DISABLE_GL
