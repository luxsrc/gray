// Copyright (C) 2012,2013 Chi-kwan Chan
// Copyright (C) 2012,2013 Steward Observatory
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

#define VERTEX_POINTER_OFFSET 1
#define COLOR_POINTER_OFFSET  6

#define R_MAP                           \
  r.w = gl_Vertex.x * sin(gl_Vertex.y); \
  r.x = r.w         * cos(gl_Vertex.z); \
  r.y = r.w         * sin(gl_Vertex.z); \
  r.z = gl_Vertex.x * cos(gl_Vertex.y); \
  r.w = 1.0;

#define C_MAP                                              \
  float A   = 0.999;                                       \
  float S   = sin(gl_Vertex.y);                            \
  float R   = gl_Vertex.x;                                 \
  float B   = gl_Color.z;                                  \
  float Sum = R * R + A * A;                               \
  float Tmp = 2.0 * R / (Sum - A * A * S * S);             \
  float G00 = Tmp - 1.0;                                   \
  float G30 = - A * Tmp * S * S;                           \
  float G33 = (Sum - A * G30) * S * S;                     \
  float Kt  = - (G33 + B * G30) / (G33 * G00 - G30 * G30); \
  gl_FrontColor.x = 0.0 + 0.2 * Kt;                        \
  gl_FrontColor.y = 0.1 * abs(B);                          \
  gl_FrontColor.z = 1.0 - 0.2 * Kt;                        \
  gl_FrontColor.w = 1.0;
