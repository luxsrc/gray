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

#ifndef VIS_H
#define VIS_H

#include <cuda_gl_interop.h> // OpenGL interoperability runtime API

#define GL_VERTEX_PROGRAM_POINT_SIZE_NV 0x8642

typedef struct {
  float x, y, z;
  float r, g, b;
} Point;

#include <map.cu>

#endif // VIS_H
