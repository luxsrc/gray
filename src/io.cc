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
#include <cstdio>

void dump(Data &data)
{
  debug("dump(*%p)\n", &data);

  static size_t frame = 0;

  const double t = global::t;
  const size_t n = data;
  const size_t m = NVAR;
  const void  *h = data.host();

  char name[256];
  snprintf(name, sizeof(name), "%04zu.raw", frame++);

  FILE *file = fopen(name, "wb");
  fwrite(&t, sizeof(double), 1, file);
  fwrite(&m, sizeof(size_t), 1, file);
  fwrite(&n, sizeof(size_t), 1, file);
  fwrite( h, sizeof(State),  n, file);
  fclose(file);
}
