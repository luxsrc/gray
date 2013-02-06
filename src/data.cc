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

#include "geode.h"
#include <cstdlib>
#include <cuda_gl_interop.h> // OpenGL interoperability runtime API

Data::Data(size_t n_input)
{
  const size_t sz = sizeof(State) * (n = n_input);
  cudaError_t err = cudaErrorMemoryAllocation; // assume we will have problem

#ifndef DISABLE_GL
  glGenBuffers(1, &vbo); // when OpenGL is enabled, we use
                         // glBufferData() to allocate device memory
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sz, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  if(GL_NO_ERROR == glGetError())
    err = cudaGraphicsGLRegisterBuffer(&res, vbo,
                                       cudaGraphicsMapFlagsWriteDiscard);
  mapped = 0; // hence, we will need to map the device memory before using it
#else
  err = cudaMalloc((void **)&res, sz); // when OpenGL is disabled, we use
                                       // cudaMalloc() to get device memory
  mapped = 1; // hence, there's no need to map between OpenGL and CUDA
#endif
  if(cudaSuccess != err)
    error("Data::Data(): fail to allocate device memory\n");

  if(!(buf = (State *)malloc(sz)))
    error("Data::Data(): fail to allocate host memory\n");
}

Data::~Data()
{
  cudaError_t err;

#ifndef DISABLE_GL
  err = cudaGraphicsUnregisterResource(res);
  glDeleteBuffers(1, &vbo);
#else
  err = cudaFree((void *)res);
#endif
  if(buf)
    free(buf);

#ifndef DISABLE_GL
  if(cudaSuccess != err || GL_NO_ERROR != glGetError())
#else
  if(cudaSuccess != err)
#endif
    error("Data::~Data(): fail to free device memory\n");
}
