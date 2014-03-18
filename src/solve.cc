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

static double ms    = 40; // assume 25 fps
static size_t count = 0;
static size_t delta = 1024;
const  size_t limit = 1048576;

bool Data::solve(double dt)
{
  debug("Data::evlove(%g)\n", dt);
  cudaError_t err;

  do {
    if(ms < 20 && delta < limit) delta *= 2;
    if(ms > 80 && delta > 1    ) delta /= 2;

    double dt_sub;
    if(count + delta < limit) {
      dt_sub = dt * delta / limit;
      count += delta;
    } else {
      dt_sub = dt * (limit - count) / limit;
      count  = 0;
    }

    if(dt_sub != 0.0) {
      err = cudaEventRecord(time0, 0);
      if(cudaSuccess != err)
	error("Data::solve(): fail to record event [%s]\n",
              cudaGetErrorString(err));

      err = evolve(dt_sub);
      if(cudaSuccess != err)
	error("Data::solve(): fail to launch kernel [%s]\n",
	      cudaGetErrorString(err));

      err = cudaEventRecord(time1, 0);
      if(cudaSuccess != err)
	error("Data::solve(): fail to record event [%s]\n",
              cudaGetErrorString(err));

      err = cudaEventSynchronize(time1);
      if(cudaSuccess == err) {
	float fms;
	err = cudaEventElapsedTime(&fms, time0, time1);
	ms  = fms;
      }
      if(cudaSuccess != err)
	error("Data::solve(): fail to obtain elapsed time [%s]\n",
	      cudaGetErrorString(err));

      err = cudaMemcpy(count_buf, count_res, sizeof(size_t) * n,
                       cudaMemcpyDeviceToHost);
      if(cudaSuccess != err)
        error("Data::solve(): fail to copy memory from device to host [%s]\n",
	      cudaGetErrorString(err));

      double actual = 0, peak = 0;
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

      if(actual)
	print("t =%7.2f; "
	      "%.0f ms/%.0f steps ~%7.2f Gflops (%.2f%%), %7.2fGB/s\n",
	      t,
	      ms,
	      actual,
	      1e-6 * scheme::flop() * actual / ms,
	      100 * actual / peak,
	      1e-6 * (24 * sizeof(real) * actual + scheme::rwsz() * n) / ms);
    }

#ifdef ENABLE_GL
#if defined(ENABLE_PRIME) || defined(ENABLE_LEAP)
    vis::sense();
#endif
    show();
    glfwSwapBuffers(vis::window);
    glfwPollEvents();
    if(glfwWindowShouldClose(vis::window))
      return false;
#endif
  } while(count);

  return true;
}
