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
static Data *d = NULL;

static void idle(void)
{
  static size_t count = 0;
  static size_t delta = 32;
  const  size_t limit = 1024;

  if(global::dt_dump != 0.0) {
    double ms;
    if(count + delta < limit) {
      ms = evolve(*d, global::dt_dump * delta / limit);
      if(0 == ms) return;
      count += delta;
    } else {
      ms = evolve(*d, global::dt_dump * (limit - count) / limit);
      if(0 == ms) return;
      count = 0;
#ifdef DUMP
      dump(*d);
#endif
    }
    if(ms < 20 && delta < limit) delta *= 2;
    if(ms > 80 && delta > 1    ) delta /= 2;
  }

#ifndef DISABLE_NITE
  sense();
#endif

  glutPostRedisplay();
}

int solve(Data &data)
{
  debug("solve(*%p)\n", &data);

  d = &data;
  glutIdleFunc(idle);

#ifdef DUMP
  dump(data);
#endif
  glutMainLoop();

  return 0;
}
#else
int solve(Data &data)
{
  debug("solve(*%p)\n", &data);

#ifdef DUMP
  do dump(data);
#endif
  while(0 < evolve(data, global::dt_dump));
#ifndef DUMP
  dump(data);
#endif

  return 0;
}
#endif
