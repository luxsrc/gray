// Copyright (C) 2012 Chi-kwan Chan
// Copyright (C) 2012 Steward Observatory
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

static __device__ void copy(real *dst, real *src, const size_t n)
{
  for(int i = 0, j = threadIdx.x; i < NVAR; ++i, j += blockDim.x)
    if(j < n) dst[j] = src[j];
}

static __global__ void driver(State *state, size_t *count, size_t n,
                              real t, real target)
{
  extern __shared__ State shared[]; // dynamic shared variable

  n     -= blockIdx.x * blockDim.x;
  state += blockIdx.x * blockDim.x;
  count += blockIdx.x * blockDim.x;

  copy((real *)shared, (real *)state, NVAR * n);
  __syncthreads();

  if(threadIdx.x < n) {
    State &s = shared[threadIdx.x];
    size_t c = 0;

    if(t < target)
      while(GET_TIME < target) {
        const real dt = scheme(s, t, target - t);
        if(0 == dt) break;
        t += dt;
        c += 1;
      }
    else
      while(GET_TIME > target) {
        const real dt = scheme(s, t, target - t);
        if(0 == dt) break;
        t += dt;
        c += 1;
      }

    count[threadIdx.x] = c;
  }

  __syncthreads();
  copy((real *)state, (real *)shared, NVAR * n);
}

static double rwsz(void)
{
  return 2 * sizeof(State) + sizeof(size_t);
}
