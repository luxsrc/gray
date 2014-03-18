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
#ifdef ENABLE_GL
#  include <cuda_gl_interop.h> // OpenGL interoperability runtime API
#endif

#define MAP_POINTER cudaGraphicsResourceGetMappedPointer // for readability

State *Data::device()
{
  debug("Data::device()\n");
#ifdef ENABLE_GL
  cudaError_t err;

  if(!mapped) {
    if(cudaSuccess != (err = cudaGraphicsMapResources(1, &res, 0)))
      error("Data::device(): fail to map OpenGL resource [%s]\n",
            cudaGetErrorString(err));
    mapped = true;
  }

  State *head = NULL;
  size_t size = 0;
  if(cudaSuccess != (err = MAP_POINTER((void **)&head, &size, res)) ||
     cudaSuccess != (err = cudaDeviceSynchronize()))
    error("Data::device(): fail to get pointer for mapped resource [%s]\n",
          cudaGetErrorString(err));

  return head;
#else
  return res;
#endif
}

State *Data::host()
{
  debug("Data::host()\n");
  cudaError_t err;

  if(cudaSuccess != (err = dtoh()))
    error("Data::device(): fail to copy device memory to host [%s]\n",
          cudaGetErrorString(err));

  return buf;
}

cudaError_t Data::deactivate()
{
  debug("Data::deactivate()\n");
  cudaError_t err = cudaSuccess; // ensure initialization

#ifdef ENABLE_GL
  if(mapped) {
    err = cudaGraphicsUnmapResources(1, &res, 0);
    mapped = false;
  }
#else
  // do nothing
#endif
  return err;
}
