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

#include "gray.h"
#include <cstdlib>

void Data::snapshot(const char *format)
{
  if(format && *format)
    debug("Data::snapshot(\"%s\")\n", format);
  else
    return;

  char name[1024];
  static int frame = 0;
  snprintf(name, sizeof(name), format, frame++);

  FILE *file = fopen(name, "wb");
  const void *h = host();
  fwrite(&t, sizeof(double), 1, file);
  fwrite(&m, sizeof(size_t), 1, file);
  fwrite(&n, sizeof(size_t), 1, file);
  fwrite( h, sizeof(State),  n, file);
  fclose(file);
}

void Data::output(const char *name, const Para &para)
{
  if(name && *name)
    debug("Data::output(\"%s\")\n", name);
  else
    return;

  FILE *file = fopen(name, "w");
  if(file) {
    output(host(), &para.buf, file);
    fclose(file);
  } else
    error("Data::output(): fail to create file \"%s\"\n", name);
}
