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

#include "gray.h"

#ifndef DISABLE_GL
#include <cmath>

static unsigned char *mkimg(int n)
{
  unsigned char *img = new unsigned char[4 * n * n];

  for(int h = 0, i = 0; i < n; ++i) {
    double x  = 2 * (i + 0.5) / n - 1;
    double x2 = x * x;
    for(int j = 0; j < n; ++j, h += 4) {
      double y  = 2 * (j + 0.5) / n - 1;
      double r2 = x2 + y * y;
      if(r2 > 1) r2 = 1;
      img[h] = img[h+1] = img[h+2] = img[h+3] =
        255 * ((2 * sqrt(r2) - 3) * r2 + 1);
    }
  }

  return img;
}

void mktexture(GLuint texture[1])
{
  glGenTextures(1, texture);

  glBindTexture(GL_TEXTURE_2D, texture[0]);
  glTexParameteri(GL_TEXTURE_2D,
                  GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
  glTexParameteri(GL_TEXTURE_2D,
                  GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D,
                  GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 64, 64, 0,
               GL_RGBA, GL_UNSIGNED_BYTE, mkimg(64));

  glActiveTextureARB(GL_TEXTURE0_ARB);

  glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);

  if(GL_NO_ERROR != glGetError())
    error("mktexture(): fail to make texture\n");
}

#endif // !DISABLE_GL
