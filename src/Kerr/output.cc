// Copyright (C) 2014 Chi-kwan Chan
// Copyright (C) 2014 Steward Observatory
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
#include <cstdio>

double image_size = 0;

void Data::output(const State *h, FILE *file)
{
  float *I = (float *)malloc(sizeof(float) * n * N_NU);
  if(!I)
    error("Data::output(): fail to allocate buffer\n");

  double total[N_NU] = {};
  for(int i = 0; i < N_NU; ++i) {
    total[i] = 0;
    for(size_t j = 0; j < n; ++j) {
      const real tmp = h[j].I[i];
      I[i*n+j]  = tmp; // real to float
      total[i] += tmp;
    }
  }

  fprintf(file, "%15.9e\t", image_size);
  for(int i = 0; i < N_NU-1; ++i)
    fprintf(file, "%15.9e\t", total[i] / n);
  fprintf(file, "%15.9e\n", total[N_NU-1] / n);

  fwrite(&n, sizeof(size_t), 1,        file);
  fwrite( I, sizeof(float),  n * N_NU, file);

  free(I);
}
