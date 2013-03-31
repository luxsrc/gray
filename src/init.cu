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
#include "harm.h"
#include <cstdlib>
#include <para.h>
#include <ic.h>

static __global__ void kernel(State *s, const size_t n, const real t)
{
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < n)
    s[i] = ic(i, n, t);
}

void init(Data &data)
{
  debug("init(*%p)\n", &data);

  const size_t n   = data;
  const size_t bsz = 64;
  const size_t gsz = (n - 1) / bsz + 1;

  State *s = data.device();
  kernel<<<gsz, bsz>>>(s, n, global::t);
  cudaError_t err = cudaDeviceSynchronize();
  data.deactivate();

  if(cudaSuccess != err)
    error("init(): fail to launch kernel; %s\n", cudaGetErrorString(err));
}

bool init_config(const char *arg)
{
  debug("init_config(""%s"")\n", arg);

  return config(arg[0], atof(arg + 2));
}
