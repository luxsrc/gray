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

Data::Data(size_t n_input)
{
  debug("Data::Data(%zu)\n", n_input);

  n   = n_input;
  m   = NVAR;
  bsz = 64;

  const size_t sz = sizeof(State) * n;
  cudaError_t err;
#ifdef ENABLE_GL
  glGenBuffers(1, &vbo); // when ENABLE_GL is enabled, we use
                         // glBufferData() to allocate device memory
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sz, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  if(GL_NO_ERROR != glGetError())
    error("Data::Data(): fail to generate or bind vertex buffer object\n");

  err = cudaGraphicsGLRegisterBuffer(&res, vbo,
                                     cudaGraphicsMapFlagsWriteDiscard);
  mapped = false; // hence, we need to map the device memory before using it
#else
  err = cudaMalloc((void **)&res, sz); // when ENABLE_GL is disabled, we use
                                       // cudaMalloc() to get device memory
  mapped = true; // hence, the memory is "always" mapped
#endif
  if(cudaSuccess != err)
    error("Data::Data(): fail to allocate device memory [%s]\n",
          cudaGetErrorString(err));

  if(!(buf = (State *)malloc(sz)))
    error("Data::Data(): fail to allocate host memory\n");
}

Data::~Data()
{
  debug("Data::~Data()\n");

  if(buf)
    free(buf);

  cudaError_t err;
#ifdef ENABLE_GL
  err = cudaGraphicsUnregisterResource(res);
  glDeleteBuffers(1, &vbo); // try deleting even if res is not unregistered
  if(cudaSuccess == err && GL_NO_ERROR != glGetError())
    err = cudaErrorUnknown;
#else
  err = cudaFree((void *)res);
#endif
  if(cudaSuccess != err)
    error("Data::~Data(): fail to free device memory [%s]\n",
          cudaGetErrorString(err));
}
