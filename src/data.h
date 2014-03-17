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

#ifndef DATA_H
#define DATA_H

class Data {
  size_t n, m, gsz, bsz;
#ifdef ENABLE_GL
  GLuint vbo;
  struct cudaGraphicsResource *res;
#else
  State *res; // device resource
#endif
  State *buf; // host buffer
  bool   mapped;

  cudaError_t dtoh();
  cudaError_t htod();

 public:
  Data(size_t = 65536);
  ~Data();

  operator size_t() { return n; }
#ifdef ENABLE_GL
  operator GLuint() { return vbo; }
#endif

  cudaError_t init  (double);
  cudaError_t evolve(double, double);

  State *device();
  State *host();
  void   deactivate();

  void dump(const char *, double);
  void spec(const char *);
};

#endif // DATA_H
