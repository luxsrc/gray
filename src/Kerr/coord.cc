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
  Coord *data = NULL;
  FILE  *file = fopen(name, "r");

  if(!file)
    error("ERROR: fail to open file \"%s\".\n", name);
  else {
    double Gamma, a_spin;
    fseek(file, 12, SEEK_CUR);
    fread(&c.nr,     sizeof(size_t), 1, file);
    fread(&c.ntheta, sizeof(size_t), 1, file);
    fread(&c.nphi,   sizeof(size_t), 1, file);
    fseek(file, 56, SEEK_CUR);
    fread(&Gamma,    sizeof(double), 1, file);
    fread(&a_spin,   sizeof(double), 1, file);
    fseek(file, 52, SEEK_CUR);
    c.Gamma  = Gamma;
    c.a_spin = a_spin;

    size_t count = c.nr * c.ntheta;
    Coord *host;
    if(!(host = (Coord *)malloc(sizeof(Coord) * count)))
      error("ERROR: fail to allocate host memory\n");
    else if(cudaSuccess != cudaMalloc((void **)&data,sizeof(Coord) * count)) {
      free(host);
      error("ERROR: fail to allocate device memory\n");
    } else {
      for(size_t i = 0; i < count; ++i) {
        double in[16];

        fseek(file, 4, SEEK_CUR);

        if(i == 0) {
          double temp;
          fseek(file, 3  * sizeof(size_t), SEEK_CUR);
          fread(&temp, sizeof(double), 1, file);
          fseek(file, 21 * sizeof(double), SEEK_CUR);
          c.lnrmin = temp;
        } else if(i == count-1) {
          double temp;
          fseek(file, 3  * sizeof(size_t), SEEK_CUR);
          fread(&temp, sizeof(double), 1, file);
          fseek(file, 21 * sizeof(double), SEEK_CUR);
          c.lnrmax = temp;
        } else
          fseek(file, 3 * sizeof(size_t) + 22 * sizeof(double), SEEK_CUR);

        fread(in, sizeof(double), 16, file);
        for(size_t j = 0; j < 16; ++j)
          (&(host[i].gcov[0][0]))[j] = in[j];

        fseek(file, 5 * sizeof(double), SEEK_CUR);
        fread(in, sizeof(double), 16, file);
        for(size_t j = 0; j < 16; ++j)
          (&(host[i].dxdxp[0][0]))[j] = in[j];

        fseek(file, 52, SEEK_CUR);
      }

      cudaError_t err =
        cudaMemcpy((void **)data, (void **)host,
                   sizeof(Coord) * count, cudaMemcpyHostToDevice);
      free(host);

      if(cudaSuccess != err) {
        cudaFree(data);
        data = NULL;
        error("ERROR: fail to copy data from host to device\n");
      }
    }

    fclose(file);
  }

  print("Data size = %zu x %zu x %zu\n"
        "Gamma = %g, spin parameter a = %g, rmin = %g, rmax = %g\n",
        c.nr, c.ntheta, c.nphi,
        c.Gamma, c.a_spin, exp(c.lnrmin), exp(c.lnrmax));

  return data;
}
