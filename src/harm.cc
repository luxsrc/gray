// Copyright (C) 2013 Chi-kwan Chan
// Copyright (C) 2013 Steward Observatory
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
#include "harm.h"
#include <cstdlib>
#include <cstring>
#include <cmath>

namespace harm {
  double t = 0, R0 = 2;
  size_t n1 = 0, n2 = 0, n3 = 0;
  real   lnrmin = 0, lnrmax = 0;
  Coord *coord = NULL;
  Field *field = NULL;
}

Coord *load_coord(const char *name)
{
  using namespace harm;

  double Gamma  = 4.0 / 3.0;
  double a_spin = 0.999;
  Coord *data   = NULL;
  FILE  *file   = fopen(name, "r");
  if(!file) {
    char upname[256] = "../";
    strcat(upname, name);
    file = fopen(upname, "r");
  }

  if(!file)
    error("ERROR: fail to open file \"%s\".\n", name);
  else {
    fseek(file, 12, SEEK_CUR);
    fread(&n1, sizeof(size_t), 1, file);
    fread(&n2, sizeof(size_t), 1, file);
    fread(&n3, sizeof(size_t), 1, file);
    fseek(file, 56, SEEK_CUR);
    fread(&Gamma,  sizeof(double), 1, file);
    fread(&a_spin, sizeof(double), 1, file);
    fread(&R0,     sizeof(double), 1, file);
    fseek(file, 44, SEEK_CUR);

    size_t count = n1 * n2;
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
          harm::lnrmin = temp;
        } else if(i == count-1) {
          double temp;
          fseek(file, 3  * sizeof(size_t), SEEK_CUR);
          fread(&temp, sizeof(double), 1, file);
          fseek(file, 21 * sizeof(double), SEEK_CUR);
          harm::lnrmax = temp;
        } else
          fseek(file, 3 * sizeof(size_t) + 22 * sizeof(double), SEEK_CUR);

        fread(in, 16 * sizeof(double), 1, file);
        for(size_t j = 0; j < 16; ++j)
          (&(host[i].gcov[0][0]))[j] = in[j];

        fseek(file, 5 * sizeof(double), SEEK_CUR);
        fread(in, 16 * sizeof(double), 1, file);
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

  if(!init_config('G', Gamma) || !prob_config('G', Gamma))
    error("load_coord(): fail to set Gamma\n");
  if(!init_config('a', a_spin) || !prob_config('a', a_spin))
    error("load_coord(): fail to set a_spin\n");

  print("Gamma = %g, spin parameter a = %g, rmin = %g, rmax = %g\n",
        Gamma, a_spin, exp(harm::lnrmin), exp(harm::lnrmax));

  return data;
}

Field *load_field(const char *name)
{
  using namespace harm;

  real max_den = 0, max_eng = 0, max_T = 0, max_v = 0, max_B = 0;

  Field *data = NULL;
  FILE  *file = fopen(name, "r");

  if(!file)
    error("ERROR: fail to open file \"%s\".\n", name);
  else {
    fscanf(file, "%lf %zu %zu %zu", &t, &n1, &n2, &n3);
    while('\n' != fgetc(file));

    size_t count = n1 * n2 * n3;
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

  print("Data size = %zu x %zu x %zu\n", n1, n2, n3);
  print("Maximum density        = %g\n"
        "Maximum energy         = %g\n"
        "Maximum temperature    = %g\n"
        "Maximum speed          = %g\n"
        "Maximum magnetic field = %g\n",
        max_den, max_eng, max_T, max_v, max_B);

  return data;
}
