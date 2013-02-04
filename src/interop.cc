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
#include <cuda_gl_interop.h> // OpenGL interoperability runtime API

State *Data::device()
{
#ifndef DISABLE_GL
  State *head = NULL;
  size_t size = 0;

  if(cudaSuccess != cudaGraphicsMapResources(1, &res, 0))
    error("Data::device(): fail to map OpenGL resource\n");

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
  return cudaSuccess == d2h() ? buf : NULL;
}

void Data::deactivate()
{
#ifndef DISABLE_GL
  if(cudaSuccess != cudaGraphicsUnmapResources(1, &res, 0))
    error("Data::device(): fail to unmap OpenGL resource\n");
#else
  // do nothing
#endif
}
