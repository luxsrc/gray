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

#ifndef DISABLE_GL
#  include <cuda_gl_interop.h> // OpenGL interoperability runtime API
#endif

State *Data::device()
{
  debug("Data::device()\n");

#ifndef DISABLE_GL
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

  return cudaSuccess == d2h() ? buf : NULL;
}

void Data::deactivate()
{
  debug("Data::deactivate()\n");

#ifndef DISABLE_GL
  if(mapped) {
    if(cudaSuccess != cudaGraphicsUnmapResources(1, &res, 0))
      error("Data::deactivate(): fail to unmap OpenGL resource\n");
    mapped = false;
  }
#else
  // do nothing
#endif
}
