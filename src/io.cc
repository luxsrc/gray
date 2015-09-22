// Copyright (C) 2012--2015 Chi-kwan Chan & Lia Medeiros
// Copyright (C) 2012--2015 Steward Observatory
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
#include <cstring>

typedef struct {
  size_t c, n;
  Point *p;
} Log;

static Log *rlog = NULL; // stands for ray-log

extern float *rays2grid(const size_t, const size_t, const Log *);

void Data::snapshot()
{
  if(!rlog) {
    rlog = (Log *)malloc(sizeof(Log) * n);
    for(size_t i = 0; i < n; ++i) {
      rlog[i].c = 0;
      rlog[i].n = 256;
      rlog[i].p = (Point *)malloc(sizeof(Point) * 256);
    }
  }

  const State *s = host();
  for(size_t i = 0; i < n; ++i) {
    size_t c = rlog[i].c;
    if(c && rlog[i].p[c-1].t == s[i].t)
      continue;

    if(c == rlog[i].n) {
      rlog[i].p = (Point *)realloc(rlog[i].p,
				   sizeof(Point) * (rlog[i].n += 256));
      if(!rlog[i].p)
	error("NOT ENOUGH MEMORY!!!\n");
    }

    point(rlog[i].p+c, s+i);
    ++(rlog[i].c);
  }
}

void Data::output(const Para &para,
		  const char *imgs, const char *rays, const char *grid)
{
  if(imgs && *imgs) {
    debug("Data::output(...): write images to \"%s\"\n", imgs);

    FILE *file = fopen(imgs, "wb");
    if(file) {
      output(host(), &para.buf, file);
      fclose(file);
    } else
      error("Data::output(): fail to create file \"%s\"\n", imgs);
  }

  if(rays && *rays && rlog) {
    debug("Data::output(...): write all rays to \"%s\"\n", rays);

    FILE *file = fopen(rays, "wb");
    if(file) {
      fwrite(&n, sizeof(size_t), 1, file);
      size_t m = sizeof(Point) / sizeof(real);
      fwrite(&m, sizeof(size_t), 1, file);
      for(size_t i = 0; i < n; ++i) {
	size_t c = rlog[i].c;
	fwrite(&c,        sizeof(size_t), 1, file);
	fwrite(rlog[i].p, sizeof(Point),  c, file);
      }
      fclose(file);
    } else
      error("Data::output(): fail to create file \"%s\"\n", rays);
  }

  if(grid && *grid && rlog) {
    debug("Data::output(...): write source grid to \"%s\"\n", grid);

    FILE *file = fopen(grid, "wb");
    if(file) {
      size_t side = 512;
      fwrite(&side, sizeof(size_t), 1, file);
      size_t n_nu = N_NU;
      fwrite(&n_nu, sizeof(size_t), 1, file);
      float *grid = rays2grid(side, n, rlog);
      fwrite(grid, sizeof(float),  side * side * side * n_nu, file);
      free(grid);
    } else
      error("Data::output(): fail to create file \"%s\"\n", grid);
  }

  if(rlog) {
    for(size_t i = 0; i < n; ++i)
      free(rlog[i].p);
    free(rlog);
    rlog = NULL;
  }
}
