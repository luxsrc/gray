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

  print("Data size = %zu x %zu x %zu\n",
        c.nr, c.ntheta, c.nphi);
  print("Gamma = %g, spin parameter a = %g, rmin = %g, rmax = %g\n",
        c.Gamma, c.a_spin, exp(c.lnrmin), exp(c.lnrmax));

  return data;
}

Field *harm::load_field(Const &c, const char *name)
{
  real max_den = 0, max_eng = 0, max_T = 0, max_v = 0, max_B = 0;

  Field *data = NULL;
  FILE  *file = fopen(name, "r");

  if(!file)
    error("ERROR: fail to open file \"%s\".\n", name);
  else {
    double time;
    size_t n1, n2, n3;
    fscanf(file, "%lf %zu %zu %zu", &time, &n1, &n2, &n3);
    while('\n' != fgetc(file));

    if(n1 != c.nr || n2 != c.ntheta || n3 != c.nphi)
      return NULL;

    size_t count = c.nr * c.ntheta * c.nphi;
    Field *host;
    if(!(host = (Field *)malloc(sizeof(Field) * count)))
      error("ERROR: fail to allocate host memory\n");
    else if(cudaSuccess != cudaMalloc((void **)&data, sizeof(Field) * count)) {
      free(host);
      error("ERROR: fail to allocate device memory\n");
    } else {
      fread(host, sizeof(Field), count, file);

      for(size_t i = 0; i < count; ++i) {
        const real v = sqrt(host[i].v1 * host[i].v1 +
                            host[i].v2 * host[i].v2 +
                            host[i].v3 * host[i].v3);
        const real B = sqrt(host[i].B1 * host[i].B1 +
                            host[i].B2 * host[i].B2 +
                            host[i].B3 * host[i].B3);
        const real T = host[i].u / host[i].rho;

        if(max_den < host[i].rho) max_den = host[i].rho;
        if(max_eng < host[i].u  ) max_eng = host[i].u;
        if(max_T   < T          ) max_T   = T;
        if(max_v   < v          ) max_v   = v;
        if(max_B   < B          ) max_B   = B;
      }

      cudaError_t err =
        cudaMemcpy((void **)data, (void **)host,
                   sizeof(Field) * count, cudaMemcpyHostToDevice);
      free(host);

      if(cudaSuccess != err) {
        cudaFree(data);
        data = NULL;
        error("ERROR: fail to copy data from host to device\n");
      }
    }

    fclose(file);
  }

  print("Maximum density        = %g\n"
        "Maximum energy         = %g\n"
        "Maximum temperature    = %g\n"
        "Maximum speed          = %g\n"
        "Maximum magnetic field = %g\n",
        max_den, max_eng, max_T, max_v, max_B);

  return data;
}
