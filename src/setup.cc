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
#include <cstdlib>
#include <cmath>
#include <algorithm>

namespace global {
  cudaEvent_t c0, c1;

#ifdef DT_DUMP
  double dt_dump = DT_DUMP;
#else
  double dt_dump = 1.0;
#endif

  double t = 0.0;
  size_t n = 65536;

  Data     *d = NULL;
  unsigned *p = NULL;
}

static void cleanup(void)
{
  using namespace global;

  if(p) {
    cudaFree(p);
    p = NULL;
  }
  if(d) {
    delete d;
    d = NULL;
  }

  cudaEventDestroy(c1);
  cudaEventDestroy(c0);
}

#include <init.cc>

int setup(int &argc, char **argv)
{
  using namespace global;

  cudaEventCreate(&c0);
  cudaEventCreate(&c1);

#ifndef DISABLE_GL
  glutInit(&argc, argv);
  glutInitWindowSize(512, 512);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  int w = glutCreateWindow(argv[0]);
#endif

  if(argc > 1) n = std::max(atoi(argv[1]), 1);
  if(argc > 2) dt_dump = atof(argv[2]);

  atexit(cleanup);

  d = new Data(n, init);
  cudaMalloc((void **)&p, sizeof(unsigned) * n);

#ifndef DISABLE_GL
  vis((GLuint)*d);
  return w;
#else
  return 0;
#endif
}
