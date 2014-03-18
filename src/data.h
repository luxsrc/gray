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

  size_t *count_res;
  size_t *count_buf;

  cudaEvent_t time0;
  cudaEvent_t time1;

#ifdef ENABLE_GL
  cudaError_t setup(GLuint *, size_t);// implemented in "src/optional/setup.cc"
  cudaError_t cleanup(GLuint *);      // implemented in "src/optional/setup.cc"
#endif
  State      *device();       // implemented in "src/interop.cc"
  State      *host();         // implemented in "src/interop.cc"
  cudaError_t deactivate();   // implemented in "src/interop.cc"
  cudaError_t dtoh();         // implemented in "src/memcpy.cc"
  cudaError_t htod();         // implemented in "src/memcpy.cc"
  cudaError_t sync(size_t *); // implemented in "src/core.cu"

 public:
  Data(size_t); // implemented in "src/data.cc"
  ~Data();      // implemented in "src/data.cc"

  double t;

  cudaError_t init  (double); // implemented in "src/core.cu"
  cudaError_t evolve(double); // implemented in "src/core.cu"

  size_t solve(double, float &, float &, float &);

  void dump(const char *); // implemented in "src/io.cc"
  void spec(const char *); // implemented in "src/io.cc"
#ifdef ENABLE_GL
  int  show();             // implemented in "src/optional/vis.cc"
#endif
};

#endif // DATA_H
