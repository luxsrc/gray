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

#include "../gray.h"

namespace global {
  int    width = 512, height = 512;
  float  ratio = 1, ax = 330, ly = -70, az = 90;
  double dt_saved  = 0;
  int    shader    = 1;
  bool   draw_body = true;
}

static double last_x = 0, last_y = 0;

void resize(GLFWwindow *win, int w, int h)
{
  global::width  = w;
  global::height = h;
  global::ratio  = (double)w / (double)h;
}

void keyboard(GLFWwindow *win, int key, int code, int action, int mods)
{
  if(GLFW_RELEASE != action) return; // do nothing

  switch(key) {
  case 'q' : case 'Q' : case GLFW_KEY_ESCAPE :
    glfwSetWindowShouldClose(global::window, GL_TRUE);
    break;
  case 'f': case 'F':
    print("TODO: switch between window/fullscreen modes\n");
    break;
  case 'h': case 'H':
    global::draw_body = !global::draw_body;
    break;
  case 's': case 'S':
    if(++global::shader >= 2) global::shader = 0;
    break;
  case 'r': case 'R':
    if(global::dt_dump == 0.0)
      global::dt_saved *= -1; // fall through
    else {
      global::dt_dump *= -1;
      break;
    }
  case 'p': case 'P':
    const double temp = global::dt_saved;
    global::dt_saved = global::dt_dump;
    global::dt_dump = temp;
    break;
  }
}

void mouse(GLFWwindow *win, double x, double y)
{
  if(GLFW_PRESS == glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_RIGHT))
    global::ly -= 0.1 * (y - last_y);
  else if(GLFW_PRESS == glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT)) {
    global::ax -= 0.5 * (x - last_x);
    global::az -= 0.5 * (y - last_y);
  }

  last_x = x;
  last_y = y;
}
