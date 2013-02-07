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
#include <cstdlib>
#include <cmath>

#ifndef N_DEFAULT
#  if defined(N_ALPHA) && defined (N_BETA)
#    define N_DEFAULT (N_ALPHA * N_BETA)
#  else
#    define N_DEFAULT 1048576 // 2^20
#  endif
#endif

#ifndef DT_DUMP
#define DT_DUMP 1.0
#endif

namespace global {
  double t       = 0.0;
  double dt_dump = 0.0;
}

int main(int argc, char **argv)
{
  print("Geode: a massive parallel geodesic integrator\n");

#ifndef DISABLE_GL
  glutInit(&argc, argv);
  glutInitWindowSize(512, 512);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutCreateWindow(argv[0]);
  if(GL_NO_ERROR != glGetError())
    error("main(): fail to initialize GLUT/OpenGL\n");
#endif

  Data data(argc > 1 ? atoi(argv[1]) : N_DEFAULT);
  global::dt_dump = argc > 2 ? atof(argv[2]) : DT_DUMP;
  init(data);

#ifndef DISABLE_GL
  vis((GLuint)data, (size_t)data);
  print("\
Press 'ESC' or 'q' to quit, 'p' to pulse, 'r' to reverse the run, 's' to\n\
to turn sprites on and off, and 'f' to enter and exit full screen\n\
");
#else
  print("Press 'Ctrl C' to quit\n");
#endif

  return solve(data);
}
