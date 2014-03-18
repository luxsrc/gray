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

#include "gray.h"

#define MEMCPY(dst, src, sz) \
  cudaMemcpy(dst, src, sz, cudaMemcpyDeviceToHost) // for readability

size_t Data::solve(double dt, float &elapse, float &actual, float &peak)
{
  debug("Data::evlove(%g)\n", dt);
  cudaError_t err;

  if(cudaSuccess != (err = cudaEventRecord(time0, 0)))
    error("Data::solve(): fail to start timing [%s]\n",
          cudaGetErrorString(err));

#ifdef ENABLE_GL
  static int    direction = 1;
  static size_t delta = DELTA;
  static size_t count = LIMIT;

  double dt_sub = dt / LIMIT;
  if(count + delta < LIMIT) {
    dt_sub *= delta;
    count  += delta;
  } else if(count == LIMIT) {
    dt_sub *= delta;
    count   = delta;
  } else {
    dt_sub *= LIMIT - count;
    count   = LIMIT;
  }

  if(direction   != 0 &&
     cudaSuccess != (err = evolve(direction * dt_sub)))
#else
  if(cudaSuccess != (err = evolve(dt)))
#endif
    error("Data::solve(): fail to launch kernel [%s]\n",
          cudaGetErrorString(err));

  if(cudaSuccess != (err = cudaEventRecord(time1, 0))                   ||
     cudaSuccess != (err = cudaEventSynchronize(time1))                 ||
     cudaSuccess != (err = cudaEventElapsedTime(&elapse, time0, time1)) ||
     cudaSuccess != (err = MEMCPY(count_buf, count_res, sizeof(size_t) * n)))
    error("Data::solve(): fail to estimate performance [%s]\n",
          cudaGetErrorString(err));

  actual = 0;
  peak   = 0;
  for(size_t j = 0, h = 0; j < gsz; ++j) {
    size_t sum = 0, max = 0;
    for(size_t i = 0; i < bsz; ++i, ++h) {
      const size_t x = (h < n) ? count_buf[h] : 0;
      sum += x;
      if(max < x) max = x;
    }
    actual += sum;
    peak   += max * bsz;
  }

#ifdef ENABLE_GL
  if(elapse < 20 && delta < LIMIT) delta *= 2;
  if(elapse > 80 && delta > 1    ) delta /= 2;

  direction = show();
  glfwPollEvents();
  if(glfwWindowShouldClose(vis::window))
    return 0;

  return actual > 0.0 ? count : 0;
#else
  return actual > 0.0 ? LIMIT : 0;
#endif
}
