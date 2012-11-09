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

#include "geode.hpp"

namespace global {
  cudaEvent_t c0, c1;
  size_t n = 0;
  State *s = NULL;
}

static void cleanup(void)
{
  using namespace global;

  if(s) {
    cudaFree(s);
    s = NULL;
  }

  cudaEventDestroy(c1);
  cudaEventDestroy(c0);
}

int setup(int &argc, char **argv)
{
  using namespace global;

  cudaEventCreate(&c0);
  cudaEventCreate(&c1);

  glutInit(&argc, argv);
  glutInitWindowSize(512, 512);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

  n = 1024;
  size_t size = sizeof(State) * n;

  atexit(cleanup);
  cudaMalloc((void **)&s, size);

  State *h = new State[n];
  for(size_t i = 0; i < n; ++i) {
    h[i].x = 20.0 * ((double)rand() / RAND_MAX - 0.5);
    h[i].y = 20.0 * ((double)rand() / RAND_MAX - 0.5);
    h[i].z = 20.0 * ((double)rand() / RAND_MAX - 0.5);
    h[i].u =  2.0 * ((double)rand() / RAND_MAX - 0.5);
    h[i].v =  2.0 * ((double)rand() / RAND_MAX - 0.5);
    h[i].w =  2.0 * ((double)rand() / RAND_MAX - 0.5);
  }
  cudaMemcpy(s, h, size, cudaMemcpyHostToDevice);
  delete[] h;

  return glutCreateWindow(argv[0]);
}
