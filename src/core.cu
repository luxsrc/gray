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

static __device__ __constant__ struct Counter {size_t *er;} count = {};

cudaError_t Data::sync(size_t *p)
{
  debug("Data::sync(%p)\n", p);

  return cudaMemcpyToSymbol(count, &p, sizeof(size_t *));
}

static __device__ __constant__ Const c = {};

cudaError_t Para::sync(Const *p)
{
  debug("Para::sync(%p)\n", p);
  cudaError_t err;

  Coord *coord = p->coord; // save the host address of coord
  Field *field = p->field; // save the host address of field

  if(coord && field) {
    size_t sz;

    sz = sizeof(Coord) * p->nr * p->ntheta;
    if(cudaSuccess != (err = cudaMalloc((void **)&p->coord, sz)) ||
       cudaSuccess != (err = cudaMemcpy(p->coord, coord, sz,
                                        cudaMemcpyHostToDevice)))
      error("Para::sync(): fail to allocate device memory [%s]\n",
            cudaGetErrorString(err));

    sz = sizeof(Field) * p->nr * p->ntheta * p->nphi;
    if(cudaSuccess != (err = cudaMalloc((void **)&p->field, sz)) ||
       cudaSuccess != (err = cudaMemcpy(p->field, field, sz,
                                        cudaMemcpyHostToDevice)))
      error("Para::sync(): fail to allocate device memory [%s]\n",
            cudaGetErrorString(err));
  }
  err = cudaMemcpyToSymbol(c, p, sizeof(Const));

  p->coord = coord; // restore the host address of coord
  p->field = field; // restore the host address of field

  return err;
}

#include <ic.h> // define device function ic()

static __global__ void kernel(State *s, const size_t n, const real t)
{
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < n)
    s[i] = ic(i, n, t);
}

cudaError_t Data::init(double t0)
{
  debug("Data::init(%g)\n", t0);
  cudaError_t err;

  kernel<<<gsz, bsz>>>(device(), n, t = t0);

  err = cudaDeviceSynchronize();
  if(cudaSuccess == err)
    err = deactivate();
  return err;
}

#include <rhs.h>   // define device function rhs()
#include <getdt.h> // define device function getdt()

#define GET(s)  ((real *)&(s))[index]
#define EACH(s) for(int index = 0; index < NVAR; ++index) GET(s)
#  include <fixup.h>      // define device function fixup()
#  include "scheme/rk4.h" // define device function integrate()
#undef GET
#undef EACH

#ifdef PARTICLE_TIME
#  define GET_TIME (t = shared[threadIdx.x].PARTICLE_TIME)
#else
#  define GET_TIME t
#endif
#  include "scheme/driver.h" // define global kernel function driver()
#undef GET_TIME

cudaError_t Data::evolve(double dt)
{
  debug("Data::evolve(%g)\n", dt);
  cudaError_t err;

  const double t0 = t, t1 = (t += dt);
  driver<<<gsz, bsz, bsz * sizeof(State)>>>(device(), n, t0, t1);

  err = cudaDeviceSynchronize();
  if(cudaSuccess == err)
    err = deactivate();
  return err;
}
