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

namespace harm {
  double t  = 0, R0 = 2;
  size_t n1 = 0, n2 = 0, n3 = 0;

  Coord *coord = NULL;
  Field *field = NULL;
}

Coord *load_coord(const char *name)
{
  using namespace harm;

  double a    = 0;
  Coord *data = NULL;
  FILE  *file = fopen(name, "r");

  if(!file)
    error("ERROR: fail to open file \"%s\".\n", name);
  else {

    fseek(file, 12, SEEK_CUR);
    fread(&n1, sizeof(size_t), 1, file);
    fread(&n2, sizeof(size_t), 1, file);
    fread(&n3, sizeof(size_t), 1, file);
    fseek(file, 64, SEEK_CUR);
    fread(&a,  sizeof(double), 1, file);
    fread(&R0, sizeof(double), 1, file);
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
        fseek(file, 4, SEEK_CUR);
        fread(host + i, sizeof(Coord), 1, file);
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

  print("Harm a = %g, R0 = %g\n", a, R0);

  return data;
}

Field *load_field(const char *name)
{
  using namespace harm;

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

  print("Harm data : %zu x %zu x %zu\n", n1, n2, n3);

  return data;
}
