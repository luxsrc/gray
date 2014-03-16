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

namespace global {
  GLFWwindow *window = NULL;
  float a_spin = 0.999;
}

static GLuint shader[2], texture;

extern void mktexture(GLuint[]);
extern void mkshaders(GLuint[]);
extern void keyboard(GLFWwindow *, int, int, int, int);
extern void mouse   (GLFWwindow *, double, double);

static void error_callback(int err, const char *msg)
{
  glfwDestroyWindow(global::window);
  glfwTerminate();
  error("[GLFW] %s\n", msg);
}

void setup(int argc, char **argv)
{
  if(!glfwInit())
    error("[GLFW] fail to initialize the OpenGL Framework\n");

  global::window = glfwCreateWindow(512, 512, argv[0], NULL, NULL);
  if(!global::window) {
    glfwTerminate();
    error("[GLFW] fail to create window\n");
  }

  glfwSetErrorCallback(error_callback);
  glfwMakeContextCurrent(global::window);
  glfwSetKeyCallback(global::window, keyboard);
  glfwSetCursorPosCallback(global::window, mouse);

  glEnable(GL_DEPTH_TEST);
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE);
  if(GL_NO_ERROR != glGetError())
    error("vis(): fail to setup visualization\n");

  mkshaders(shader);
  mktexture(&texture);
}


void display(size_t n, GLuint vbo)
{
  int width, height;
  glfwGetFramebufferSize(global::window, &width, &height);

  glViewport(0, 0, width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(27.0, (float)width / height, 1.0, 1.0e6);
  glMatrixMode(GL_MODELVIEW);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
#ifdef ENABLE_PRIME
  if(draw_body) track();
#endif
  glLoadIdentity();
  glRotatef(-90, 1, 0, 0);
  glTranslatef(0, -global::ly, 0);
  glRotatef(-(global::az- 90), 1, 0, 0);
  glRotatef(-(global::ax-270), 0, 0, 1);

  // Draw wire sphere, i.e., the "black hole"
  glColor3f(0.0, 1.0, 0.0);
  glutWireSphere(1.0 + sqrt(1.0 - global::a_spin * global::a_spin), 32, 16);

  // Draw particles, i.e., photon locations
  glUseProgram(shader[global::shader]);

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
}
