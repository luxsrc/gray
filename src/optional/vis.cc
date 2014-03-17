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
#include <shader.h> // to get vertex and color pointer offsets

#ifndef VERTEX_POINTER_OFFSET
#define VERTEX_POINTER_OFFSET 0
#endif

#ifndef COLOR_POINTER_OFFSET
#define COLOR_POINTER_OFFSET  3
#endif

#define GL_VERTEX_PROGRAM_POINT_SIZE_NV 0x8642

namespace vis {
  GLFWwindow *window = NULL;
  float a_spin = 0.999;
}

static GLuint shaders[2], texture;

static void error_callback(int err, const char *msg)
{
  glfwDestroyWindow(vis::window);
  glfwTerminate();
  error("[GLFW] %s\n", msg);
}

void vis::setup(int argc, char **argv, Para &para)
{
  vis::p = &para;

  if(!glfwInit())
    error("[GLFW] fail to initialize the OpenGL Framework\n");

  vis::window = glfwCreateWindow(512, 512, argv[0], NULL, NULL);
  if(!vis::window) {
    glfwTerminate();
    error("[GLFW] fail to create window\n");
  }

  glfwSetErrorCallback     (error_callback);
  glfwSetWindowSizeCallback(vis::window, vis::resize);
  glfwSetKeyCallback       (vis::window, vis::keyboard);
  glfwSetCursorPosCallback (vis::window, vis::mouse);
  glfwMakeContextCurrent   (vis::window);

  glEnable(GL_DEPTH_TEST);
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE);
  if(GL_NO_ERROR != glGetError())
    error("vis::setup(): fail to setup visualization\n");

  vis::mkshaders(shaders);
  vis::mktexture(&texture);
}

void vis::show(size_t n, GLuint vbo)
{
  glViewport(0, 0, vis::width, vis::height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(27.0, vis::ratio, 1.0, 1.0e6);
  glMatrixMode(GL_MODELVIEW);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
#ifdef ENABLE_PRIME
  if(vis::draw_body) vis::track();
#endif
  glLoadIdentity();
  glRotatef(-90, 1, 0, 0);
  glTranslatef(0, -vis::ly, 0);
  glRotatef(-(vis::az- 90), 1, 0, 0);
  glRotatef(-(vis::ax-270), 0, 0, 1);

  // Draw wire sphere, i.e., the "black hole"
  glColor3f(0.0, 1.0, 0.0);
  glutWireSphere(1.0 + sqrt(1.0 - vis::a_spin * vis::a_spin), 32, 16);

  // Draw particles, i.e., photon locations
  glUseProgram(shaders[vis::shader]);

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
    error("vis::show(): fail to visualize simulation\n");
}
