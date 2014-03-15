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

#include "../gray.h"

#include <shader.h>

#define STR1NG(x) #x
#define STRING(x) STR1NG(x)

static GLuint compile(const char *src, GLenum type)
{
  GLuint s = glCreateShader(type);
  glShaderSource(s, 1, &src, 0);
  glCompileShader(s);
  return s;
}

void mkshaders(GLuint shader[3])
{
  shader[0] = glCreateProgram();
  glAttachShader(shader[0], compile(STRING(
    void main()
    {
      vec4 r;
  ) STRING(R_MAP) STRING(C_MAP) STRING(
      gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * r;
    }
  ), GL_VERTEX_SHADER));
  glLinkProgram(shader[0]);

  shader[1] = glCreateProgram();
  glAttachShader(shader[1], compile(STRING(
    void main()
    {
      vec4 r;
  ) STRING(R_MAP) STRING(C_MAP) STRING(
      vec4 q = gl_ModelViewMatrix * r;
      gl_Position    = gl_ProjectionMatrix * q;
      gl_PointSize   = max(2.0, 200.0 * gl_Point.size / (1.0 - q.z));
      gl_TexCoord[0] = gl_MultiTexCoord0;
    }
  ), GL_VERTEX_SHADER));
  glAttachShader(shader[1], compile(STRING(
    uniform sampler2D splatTexture;
    void main()
    {
      gl_FragColor = gl_Color
                   * texture2D(splatTexture, gl_TexCoord[0].st);
    }
  ), GL_FRAGMENT_SHADER));
  glLinkProgram(shader[1]);

  if(GL_NO_ERROR != glGetError())
    error("mkshaders(): fail to compile shader\n");
}
