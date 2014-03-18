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

#define GL_VERTEX_PROGRAM_POINT_SIZE_NV 0x8642

namespace vis {
  GLFWwindow *window = NULL;
  GLuint shaders[2];
}

static void error_callback(int err, const char *msg)
{
  glfwDestroyWindow(vis::window);
  glfwTerminate();
  error("[GLFW] %s\n", msg);
}

void Data::setup()
{
  if(!glfwInit())
    error("[GLFW] fail to initialize the OpenGL Framework\n");

  vis::window = glfwCreateWindow(512, 512, "GRay", NULL, NULL);
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

  GLuint texture;
  vis::mktexture(    &texture);
  vis::mkshaders(vis::shaders);
}
