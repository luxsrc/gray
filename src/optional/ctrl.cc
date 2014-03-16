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

#include <cstdlib>
#include <para.h>

#ifndef WIDTH
#define WIDTH 512
#endif

#ifndef HEIGHT
#define HEIGHT 512
#endif

#ifdef ENABLE_PRIME
extern void track();
#endif

namespace global {
  float ax = 330, ly = -70, az = 90;
}

static int last_x = 0, last_y = 0;
static int left   = 0, right  = 0;

static bool fullscreen = false;
static bool draw_body  = true;
static int  shader     = 1;

static void keyboard(unsigned char key, int x, int y)
{
  using namespace global;

  switch(key) {
  case 'q': case 'Q': case 27: // ESCAPE key
    exit(0);
    break;
  case 'f': case 'F':
    if((fullscreen = !fullscreen))
      glutFullScreen();
    else
      glutReshapeWindow(WIDTH, HEIGHT);
    break;
  case 'h': case 'H':
    draw_body = !draw_body;
    break;
  case 's': case 'S':
    if(++shader >= 2) shader = 0;
    break;
  case 'r': case 'R':
    if(dt_dump == 0.0)
      dt_saved *= -1; // fall through
    else {
      dt_dump *= -1;
      break;
    }
  case 'p': case 'P':
    const double temp = dt_saved;
    dt_saved = dt_dump;
    dt_dump = temp;
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
  using namespace global;

  const int dx = x - last_x; last_x = x;
  const int dy = y - last_y; last_y = y;

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
  using namespace global;

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

#ifdef ENABLE_PRIME
  if(draw_body) track();
#endif

  glLoadIdentity();
  glRotatef(-90, 1, 0, 0);
  glTranslatef(0, -ly, 0);
  glRotatef(-(az- 90), 1, 0, 0);
  glRotatef(-(ax-270), 0, 0, 1);

  return shader;
}
