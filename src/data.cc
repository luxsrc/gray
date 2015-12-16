// Copyright (C) 2012--2014 Chi-kwan Chan
// Copyright (C) 2012--2014 Steward Observatory
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
#ifdef ENABLE_GL
#  include <cuda_gl_interop.h> // OpenGL interoperability runtime API
#endif

#define REG_BUFFER    cudaGraphicsGLRegisterBuffer     // for readability
#define WRITE_DISCARD cudaGraphicsMapFlagsWriteDiscard // for readability

Data::Data(size_t n_input)
{
  debug("Data::Data(%zu)\n", n_input);
  cudaError_t err;

  n   = n_input;
  m   = NVAR;
  bsz = (sizeof(real) == sizeof(float) && N_NU < 24) ? 64 : 32;
  gsz = (n - 1) / bsz + 1;
  t   = 0.0;

  const size_t ssz = sizeof(State)  * n;
  const size_t csz = sizeof(size_t) * n;
#ifdef ENABLE_GL
  mapped = false; // we need to map the device memory before using it
  if(cudaSuccess != (err = setup(&vbo, ssz)) ||
     cudaSuccess != (err = REG_BUFFER(&res, vbo, WRITE_DISCARD)) ||
#else
  mapped = true; // the memory is "always" mapped
  if(cudaSuccess != (err = cudaMalloc((void **)&res, ssz)) ||
#endif
     cudaSuccess != (err = cudaMalloc((void **)&count_res, csz)) ||
     cudaSuccess != (err = sync(count_res)))
    error("Data::Data(): fail to allocate device memory [%s]\n",
          cudaGetErrorString(err));

  if(cudaSuccess != (err = cudaEventCreate(&time0)) ||
     cudaSuccess != (err = cudaEventCreate(&time1)))
    error("Data::Data(): fail to create timer [%s]\n",
          cudaGetErrorString(err));

  if(!(buf       = (State  *)malloc(ssz)) ||
     !(count_buf = (size_t *)malloc(csz)))
    error("Data::Data(): fail to allocate host memory\n");
}

Data::~Data()
{
  debug("Data::~Data()\n");
  cudaError_t err;

  free(count_buf);
  free(buf);
  // TODO: check errno for free() error?

  if(cudaSuccess != (err = cudaEventDestroy(time1)) ||
     cudaSuccess != (err = cudaEventDestroy(time0)))
    error("Para::~Para(): fail to destroy timer [%s]\n",
          cudaGetErrorString(err));

  if(cudaSuccess != (err = cudaFree((void *)count_res)) ||
#ifdef ENABLE_GL
     cudaSuccess != (err = cudaGraphicsUnregisterResource(res)) ||
     cudaSuccess != (err = cleanup(&vbo)))
#else
     cudaSuccess != (err = cudaFree((void *)res)))
#endif
    error("Data::~Data(): fail to free device memory [%s]\n",
          cudaGetErrorString(err));
}
