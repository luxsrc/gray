// Copyright (C) 2013,2014 Chi-kwan Chan
// Copyright (C) 2013,2014 Steward Observatory
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
#include <cmath>

Coord *harm::load_coord(Const &c, const char *name)
{
  cudaError_t err;

  FILE *file = fopen(name, "r");
  if(!file)
    error("ERROR: fail to open file \"%s\".\n", name);

  size_t count, sz;
  fseek(file, 12, SEEK_CUR);
  fread(&c.nr,     sizeof(size_t), 1, file);
  fread(&c.ntheta, sizeof(size_t), 1, file);
  fread(&c.nphi,   sizeof(size_t), 1, file);
  count = c.nr * c.ntheta;
  sz    = sizeof(Coord) * count;

  double Gamma, a_spin;
  fseek(file, 56, SEEK_CUR);
  fread(&Gamma,    sizeof(double), 1, file);
  fread(&a_spin,   sizeof(double), 1, file);
  fseek(file, 52, SEEK_CUR);
  c.Gamma  = Gamma;
  c.a_spin = a_spin;

  Coord *host;
  if(!(host = (Coord *)malloc(sz)))
    error("ERROR: fail to allocate host memory\n");
  for(size_t i = 0; i < count; ++i) {
    double in[16];

    fseek(file, 4 + 3 * sizeof(size_t) + 3 * sizeof(double), SEEK_CUR);

    fread(in, sizeof(double), 2, file);
    if(i < c.nr && i < N_R) c.r[i] = in[0];
    host[i].theta = in[1];

    fseek(file, 17 * sizeof(double), SEEK_CUR);

    fread(in, sizeof(double), 16, file);
    for(size_t j = 0; j < 16; ++j)
      (&(host[i].gcov[0][0]))[j] = in[j];

    fseek(file, 5 * sizeof(double), SEEK_CUR);

    fread(in, sizeof(double), 16, file);
    for(size_t j = 0; j < 16; ++j)
      (&(host[i].dxdxp[0][0]))[j] = in[j];

    fseek(file, 52, SEEK_CUR);
  }
  fclose(file);

  Coord *data;
  if(cudaSuccess != (err = cudaMalloc((void **)&data, sz)) ||
     cudaSuccess != (err = cudaMemcpy(data, host, sz, cudaMemcpyHostToDevice)))
    error("ERROR: fail to allocate device memory [%s]\n",
          cudaGetErrorString(err));
  free(host);

  print("Data size = %zu x %zu x %zu\n"
        "Gamma = %g, spin parameter a = %g, rmin = %g, rmax = %g\n",
        c.nr, c.ntheta, c.nphi, c.Gamma, c.a_spin, c.r[0], c.r[c.nr-1]);

  return data;
}
