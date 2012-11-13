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
static void idle(void)
{
  using namespace global;

  const size_t n_vis = 20;
  evolve(dt_dump / n_vis, 100 / n_vis);

  static size_t i = 0;
  if(++i == n_vis) {
    i = 0;
    dump();
  }
  vis();

  glutPostRedisplay();
}

int solve(void)
{
  dump();
  vis();

  glutIdleFunc(idle);
  glutMainLoop();

  return 0;
}
#else
static void mainloop(void)
{
  using namespace global;

  dump();
  for(;;) {
    evolve(dt_dump, 100);
    dump();
  }
}

int solve(void)
{
  mainloop();

  return 0;
}
#endif
