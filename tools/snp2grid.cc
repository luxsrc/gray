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

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <stdint.h>

const size_t N = 222;

typedef struct {
  float t, r, theta, phi, I;
} Point;

int main(int argc, char **argv)
{
  if(argc < 3) {
    fprintf(stderr, "usage: %s [input] [output]\n", argv[0]);
    return -1;
  }

  double *dI = (double *)malloc(sizeof(double) * N * N * N);
  for(size_t i = 0; i < N * N * N; ++i)
    dI[i] = 0.0;

  FILE *file = fopen(argv[1], "rb");
  if(!file) {
    fprintf(stderr, "input file \"%s\" not found\n", argv[1]);
    return -1;
  }

  size_t n, nvar;
  fread(&n,    sizeof(size_t), 1, file);
  fread(&nvar, sizeof(size_t), 1, file);
  if(nvar != sizeof(Point) / sizeof(float)) {
    fprintf(stderr, "Invalid data size\n");
    return -1;
  }

  size_t count;
  Point *point;
  for(size_t i = 0; i < n; ++i) {
    fread(&count, sizeof(size_t), 1, file);

    point = (Point *)malloc(sizeof(Point) * (count+2));
    fread(point+1, sizeof(Point), count, file);
    point[0].I       = point[1].I;
    point[count+1].I = point[count].I;

    printf("\rProcessing %6.2f%%", 100.0 / n * (i+1));
    fflush(stdout);

    for(size_t j = 1; j <= count; ++j) {
      double r     = point[j].r;
      double theta = point[j].theta;
      double phi   = point[j].phi;
      double R     = r * sin(theta);

      int ix = (int)((R * cos(phi)   + 0.25 * N) * 2.0 + 0.5);
      int iy = (int)((R * sin(phi)   + 0.25 * N) * 2.0 + 0.5);
      int iz = (int)((r * cos(theta) + 0.25 * N) * 2.0 + 0.5);

      if(0 < ix && ix < N &&
	 0 < iy && iy < N &&
	 0 < iz && iz < N) {
	size_t k = (ix * N + iy) * N + iz;
	dI[k] += (point[j+1].I - point[j-1].I) / 2;
      }
    }

    free(point);
  }
  printf("\n");
  fclose(file);

  double max = 0;
  for(size_t i = 0; i < N * N * N; ++i)
    if(max < dI[i])
      max = dI[i];

  printf("max = %f\n", max);

  uint8_t *data = (uint8_t *)malloc(N * N * N * 4);
  for(size_t i = 0; i < N * N * N; ++i) {
    double temp = 2 * 255 * dI[i] / max;
    data[i * 4 + 0] = 255;
    data[i * 4 + 1] = 255;
    data[i * 4 + 2] = 255;
    data[i * 4 + 3] = temp <= 255 ? temp : 255;
  }

  file = fopen(argv[2], "wb");
  fwrite(&N,   sizeof(size_t), 1, file);
  fwrite(data, 1, N * N * N * 4, file);
  fclose(file);

  free(data);
  free(dI);

  return 0;
}
