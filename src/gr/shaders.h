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

#define VERTEX_SHADER                                                   \
  void main()                                                           \
  {                                                                     \
    vec4 vert;                                                          \
    vert.w = gl_Vertex.x * sin(gl_Vertex.y);                            \
    vert.x = vert.w      * cos(gl_Vertex.z);                            \
    vert.y = vert.w      * sin(gl_Vertex.z);                            \
    vert.z = gl_Vertex.x * cos(gl_Vertex.y);                            \
    vert.w = 1.0;                                                       \
    vec3 pos_eye = vec3(gl_ModelViewMatrix * vert);                     \
    gl_PointSize = max(1.0, 500.0 * gl_Point.size / (1.0 - pos_eye.z)); \
    gl_TexCoord[0] = gl_MultiTexCoord0;                                 \
    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vert;      \
    gl_FrontColor = gl_Color;                                           \
    gl_FrontSecondaryColor = gl_SecondaryColor;                         \
  }

#define PIXEL_SHADER                                           \
  uniform sampler2D splatTexture;                              \
  void main()                                                  \
  {                                                            \
    vec4 color   = (0.6 + 0.4 * gl_Color)                      \
      * texture2D(splatTexture, gl_TexCoord[0].st);            \
    gl_FragColor = color * gl_SecondaryColor;                  \
  }
