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

State *Data::device()
{
  debug("Data::device()\n");

#ifdef ENABLE_GL
  State *head = NULL;
  size_t size = 0;

  if(!mapped) {
    if(cudaSuccess != cudaGraphicsMapResources(1, &res, 0))
      error("Data::device(): fail to map OpenGL resource\n");
    mapped = true;
  }
  if(cudaSuccess !=
     cudaGraphicsResourceGetMappedPointer((void **)&head, &size, res))
    error("Data::device(): fail to get pointer for mapped resource\n");

  return head;
#else
  return res;
#endif
}

State *Data::host()
{
  debug("Data::host()\n");

  return cudaSuccess == dtoh() ? buf : NULL;
}

cudaError_t Data::deactivate()
{
  debug("Data::deactivate()\n");

  cudaError_t err = cudaSuccess;
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
