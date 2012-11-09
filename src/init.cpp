// Copyright (C) 2012 Chi-kwan Chan
// Copyright (C) 2012 Steward Observatory
//
// This file is part of geode.
//
// Geode is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Geode is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
// or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
// License for more details.
//
// You should have received a copy of the GNU General Public License
// along with geode.  If not, see <http://www.gnu.org/licenses/>.

#include "geode.hpp"

#include <cuda_runtime_api.h>

static void *d = NULL;

static void fini(void)
{
  if(d) {
    cudaFree(d);
    d = NULL;
  }
}

State *init(size_t n)
{
  const size_t size = sizeof(State) * n;

  if(!d) {
    atexit(fini);
    cudaMalloc(&d, size);
  }

  State *h = new State[n];
  for(size_t i = 0; i < n; ++i) {
    h[i].x = 20.0 * ((double)rand() / RAND_MAX - 0.5);
    h[i].y = 20.0 * ((double)rand() / RAND_MAX - 0.5);
    h[i].z = 20.0 * ((double)rand() / RAND_MAX - 0.5);
    h[i].u =  0.5 * ((double)rand() / RAND_MAX + 1.0);
    h[i].v =  0.5 * ((double)rand() / RAND_MAX + 1.0);
    h[i].w =  0.5 * ((double)rand() / RAND_MAX + 1.0);
  }
  cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);
  delete[] h;

  return (State *)d;
}
