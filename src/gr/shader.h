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

#define VERTEX_POINTER_OFFSET 1
#define COLOR_POINTER_OFFSET  4

#define R_MAP                           \
  r.w = gl_Vertex.x * sin(gl_Vertex.y); \
  r.x = r.w         * cos(gl_Vertex.z); \
  r.y = r.w         * sin(gl_Vertex.z); \
  r.z = gl_Vertex.x * cos(gl_Vertex.y); \
  r.w = 1.0;

#define C_MAP gl_FrontColor = 0.5 * abs(gl_Color);

