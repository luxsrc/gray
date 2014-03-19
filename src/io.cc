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
#include <cstdio>

void Data::snapshot(const char *format)
{
  if(format && *format)
    debug("Data::snapshot(\"%s\")\n", format);
  else
    return;

  char name[256];
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

void Data::output(const char *name)
{
  if(name && *name)
    debug("Data::output(\"%s\")\n", name);
  else
    return;

#ifdef HARM
  const State *h = host();

  float *I = (float *)malloc(sizeof(float) * n * N_NU);
  double total[N_NU] = {};
  if(!I)
    error("ERROR: fail to allocate buffer\n");
  else
    for(int i = 0; i < N_NU; ++i) {
      total[i] = 0;
      for(size_t j = 0; j < n; ++j) {
        const real tmp = h[j].rad[i].I;
        I[i*n+j]  = tmp; // real to float
        total[i] += tmp;
      }
    }

  FILE *file = fopen(name, "w");
  for(int i = 0; i < N_NU-1; ++i) fprintf(file, "%15e ", total[i] / n);
  fprintf(file, "%15e\n", total[N_NU-1] / n);
  fwrite(&n, sizeof(size_t), 1,        file);
  fwrite( I, sizeof(float),  n * N_NU, file);
  fclose(file);

  free(I);
#endif
}
