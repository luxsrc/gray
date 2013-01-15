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

#ifndef DATA_H
#define DATA_H

class Data {
  size_t n;
#ifndef DISABLE_GL
  GLuint vbo;
  struct cudaGraphicsResource *res;
#else
  State *res; // device resource
#endif
  State *buf; // host buffer

 public:
  Data(size_t = 65536, State (*)(int) = NULL);
  ~Data();
#ifndef DISABLE_GL
  operator GLuint() { return vbo; }
#endif
  State *device();
  State *host();
  void deactivate();
};

#endif
