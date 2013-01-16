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

#ifndef DISABLE_GL
static void idle(void)
{
  using namespace global;

  static size_t count = 0;
  static size_t delta = 32;
  const  size_t limit = 1024;

  if(dt_dump != 0.0) {
    float ms;
    if(count + delta < limit) {
      ms = evolve(dt_dump * delta / limit);
      count += delta;
    } else {
      ms = evolve(dt_dump * (limit - count) / limit);
      count = 0;
      dump();
    }
    if(ms < 10 && delta < limit) delta *= 2;
    if(ms > 40 && delta > 1    ) delta /= 2;
  }

  glutPostRedisplay();
}

int solve(void)
{
  dump();

  glutIdleFunc(idle);
  glutMainLoop();

  return 0;
}
#else
int solve(void)
{
  dump();
  for(;;) {
    evolve(dt_dump);
    dump();
  }
  return 0;
}
#endif
