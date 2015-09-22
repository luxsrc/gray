// Copyright (C) 2015 Chi-kwan Chan & Lia Medeiros
// Copyright (C) 2015 Steward Observatory
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
#include <cmath>
#include <stdint.h>

typedef struct {
  size_t c, n;
  Point *p;
} Log;

float *rays2grid(const size_t s, const size_t n, const Log *l)
{
  double *dI = (double *)malloc(sizeof(double) * s * s * s * N_NU);
  for(size_t i = 0; i < s * s * s * N_NU; ++i)
    dI[i] = 0.0;

  for(size_t i = 0; i < n; ++i) {
    size_t c = l[i].c;
    Point *p = l[i].p;

    for(size_t j = 0; j < c; ++j) {
      double r     = p[j].r;
      double theta = p[j].theta;
      double phi   = p[j].phi;
      double R     = r * sin(theta);

      int ix = (int)(8.0 * R * cos(phi)   + 0.5 * (s + 1));
      int iy = (int)(8.0 * R * sin(phi)   + 0.5 * (s + 1));
      int iz = (int)(8.0 * r * cos(theta) + 0.5 * (s + 1));

      if(0 <= ix && ix < (int)s &&
	 0 <= iy && iy < (int)s &&
	 0 <= iz && iz < (int)s) {
	size_t h = (ix * s + iy) * s + iz;

	if(j == 0)
	  for(size_t k = 0; k < N_NU; ++k)
	    dI[k * s * s * s + h] += p[1].I[k] - p[0].I[k];
	else if(j < c-1)
	  for(size_t k = 0; k < N_NU; ++k)
	    dI[k * s * s * s + h] += (p[j+1].I[k] - p[j-1].I[k]) / 2;
	else
	  for(size_t k = 0; k < N_NU; ++k)
	    dI[k * s * s * s + h] += (p[c-1].I[k] - p[c-2].I[k]);
      }
    }
  }

  float *grid = (float *)malloc(sizeof(float) * s * s * s * N_NU);
  for(size_t i = 0; i < s * s * s * N_NU; ++i)
    grid[i] = dI[i];

  free(dI);

  return grid;
}
