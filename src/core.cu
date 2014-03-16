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

static __device__ __constant__ size_t *count = NULL;

#include "Kerr/harm.h"
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
  const size_t gsz = (n - 1) / global::bsz + 1;

  State *s = data.device();
  kernel<<<gsz, global::bsz>>>(s, n, global::t);
  cudaError_t err = cudaDeviceSynchronize();
  data.deactivate();

  if(cudaSuccess != err)
    error("init(): fail to launch kernel [%s]\n",
          cudaGetErrorString(err));
}

bool init_config(const char *arg)
{
  debug("init_config(""%s"")\n", arg);
  return config(arg[0], atof(arg + 2));
}

bool init_config(char flag, real val)
{
  debug("init_config(""%c=%d"")\n", flag, val);
  return config(flag, val);
}

#include <rhs.h>
#include <getdt.h>

#define GET(s)  ((real *)&(s))[index]
#define EACH(s) for(int index = 0; index < NVAR; ++index) GET(s)
#  include <fixup.h>
#  include "scheme/rk4.h"
#undef GET
#undef EACH

#ifdef PARTICLE_TIME
#  define GET_TIME (t = shared[threadIdx.x].PARTICLE_TIME)
#else
#  define GET_TIME t
#endif
#  include "scheme/driver.h"
#undef GET_TIME

static size_t *res = NULL, *buf = NULL;
static cudaEvent_t time0, time1;

static void setup(size_t n)
{
  if(cudaSuccess != cudaMalloc((void **)&res, sizeof(size_t) * n))
    error("evolve(): fail to allocate device memory\n");
  if(cudaSuccess != cudaMemcpyToSymbol(count, &res, sizeof(size_t *)))
    error("evolve(): fail to sync device memory address to constant memory\n");
  if(NULL == (buf = (size_t *)malloc(sizeof(size_t) * n)))
    error("evolve(): fail to allocate host memory\n");

#ifdef HARM
  if(cudaSuccess != cudaMemcpyToSymbol(coord, &harm::coord, sizeof(Coord*)) ||
     cudaSuccess != cudaMemcpyToSymbol(field, &harm::field, sizeof(Field*)) ||
     cudaSuccess != cudaMemcpyToSymbol(lnrmin,&harm::lnrmin,sizeof(real  )) ||
     cudaSuccess != cudaMemcpyToSymbol(lnrmax,&harm::lnrmax,sizeof(real  )) ||
     cudaSuccess != cudaMemcpyToSymbol(nr,    &harm::n1,    sizeof(int   )) ||
     cudaSuccess != cudaMemcpyToSymbol(ntheta,&harm::n2,    sizeof(int   )) ||
     cudaSuccess != cudaMemcpyToSymbol(nphi,  &harm::n3,    sizeof(int   )))
    error("evolve(): fail to copy pointer(s) to device\n");
#endif

  if(cudaSuccess != cudaEventCreate(&time0) ||
     cudaSuccess != cudaEventCreate(&time1))
    error("evolve(): fail to create timer\n");
}

static void cleanup(void)
{
  if(res) {
    cudaFree(res);
    res = NULL;
    cudaMemcpyToSymbol(count, &res, sizeof(size_t *)); // set count to NULL
  }

  if(buf) {
    free(buf);
    buf = NULL;
  }

  if(cudaSuccess != cudaEventDestroy(time1) ||
     cudaSuccess != cudaEventDestroy(time0))
    error("evolve(): fail to destroy timer\n");
}

double evolve(Data &data, double dt)
{
  debug("evolve(*%p, %g)\n", &data, dt);

  const double t = global::t;
  const size_t n = data;

  if(!res && !buf && !atexit(cleanup)) setup(n);

  const size_t gsz = (n - 1) / global::bsz + 1;

  if(cudaSuccess != cudaEventRecord(time0, 0))
    error("evolve(): fail to record event\n");

  State *s = data.device();
  driver<<<gsz, global::bsz,
                global::bsz * sizeof(State)>>>(s, n, t, global::t += dt);
  cudaError_t err = cudaDeviceSynchronize();
  data.deactivate();

  if(cudaSuccess != cudaEventRecord(time1, 0))
    error("evolve(): fail to record event\n");
  if(cudaSuccess != err)
    error("evolve(): fail to launch kernel [%s]\n",
          cudaGetErrorString(err));

  float ms;
  if(cudaSuccess != cudaEventSynchronize(time1) ||
     cudaSuccess != cudaEventElapsedTime(&ms, time0, time1))
    error("evolve(): fail to obtain elapsed time\n");

  if(cudaSuccess !=
     cudaMemcpy(buf, res, sizeof(size_t) * n, cudaMemcpyDeviceToHost))
    error("evolve(): fail to copy memory from device to host\n");

  double actual = 0, peak = 0;
  for(size_t j = 0, h = 0; j < gsz; ++j) {
    size_t sum = 0, max = 0;
    for(size_t i = 0; i < global::bsz; ++i, ++h) {
      const size_t x = (h < n) ? buf[h] : 0;
      sum += x;
      if(max < x) max = x;
    }
    actual += sum;
    peak   += max * global::bsz;
  }

  if(actual) {
    print("t =%7.2f; %.0f ms/%.0f steps ~%7.2f Gflops (%.2f%%),%7.2fGB/s\n",
          global::t, ms, actual, 1e-6 * flop() * actual / ms,
          100 * actual / peak,   1e-6 * (24 * sizeof(real) * actual +
                                         rwsz() * n) / ms); // read + write
    return ms;
  } else
    return 0;
}

bool prob_config(const char *arg)
{
  debug("prob_config(""%s"")\n", arg);

  return config(arg[0], atof(arg + 2));
}

bool prob_config(char flag, real val)
{
  debug("prob_config(""%c=%d"")\n", flag, val);

  return config(flag, val);
}
