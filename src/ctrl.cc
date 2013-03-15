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
#include <cstdlib>

static float ax = 330, az = 90, ly =-70; // view angles

static int last_x = 0, last_y = 0;
static int left   = 0, right  = 0;

static int fullscreen = 0;
static int shader     = 1;

static void keyboard(unsigned char key, int x, int y)
{
  static double dt_stored = 0.0;

  switch(key) {
  case 'q': case 'Q': case 27: // ESCAPE key
    exit(0);
    break;
  case 'f': case 'F':
    if((fullscreen = !fullscreen))
      glutFullScreen();
    else
      glutReshapeWindow(512, 512);
    break;
  case 's': case 'S':
    if(++shader >= 2) shader = 0;
    break;
  case 'r': case 'R':
    if(global::dt_dump == 0.0)
      dt_stored *= -1; // fall through
    else {
      global::dt_dump *= -1;
      break;
    }
  case 'p': case 'P':
    double temp = dt_stored;
    dt_stored = global::dt_dump;
    global::dt_dump = temp;
    break;
  }
}

static void mouse(int b, int s, int x, int y)
{
  last_x = x;
  last_y = y;

  switch(b) {
  case GLUT_LEFT_BUTTON : left  = (s == GLUT_DOWN); break;
  case GLUT_RIGHT_BUTTON: right = (s == GLUT_DOWN); break;
  }
}

static void motion(int x, int y)
{
  int dx = x - last_x; last_x = x;
  int dy = y - last_y; last_y = y;

  if(right)
    ly -= 0.1 * dy;
  else if(left) {
    az -= 0.5 * dy;
    ax -= 0.5 * dx;
  }

  glutPostRedisplay();
}

void regctrl()
{
  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
}

int getctrl()
{
  glLoadIdentity();

  glRotatef(-90, 1, 0, 0);
  glTranslatef(0, -ly, 0);
  glRotatef(-(az- 90), 1, 0, 0);
  glRotatef(-(ax-270), 0, 0, 1);

  return shader;
}

#endif // !DISABLE_GL
