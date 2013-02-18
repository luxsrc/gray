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
#include <cstdio>

void dump(Data &data)
{
  debug("dump(*%p)\n", &data);

#ifdef DUMP
  static size_t frame = 0;

  const size_t m = 181;
  const size_t n = (size_t)data / m;
  const void  *h = data.host();
  const State *s = (State *)h;

  real r[m];
  for(size_t i, j = 0; j < m; ++j) {
    for(i = 0; i < n && s[j * n + i].r < 3; ++i);
    r[j] = 1.5 + (6.0 / n) * i;
  }

  char name[256];
  snprintf(name, sizeof(name), "%04zu.raw", frame++);

  FILE *file = fopen(name, "wb");
  if(file) {
    fwrite(r, sizeof(real), m, file);
    fclose(file);
  } else
    error("dump(): fail to output to file ""%s""\n", name);
#endif
}
