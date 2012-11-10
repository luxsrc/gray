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

#include <rhs.cu>

#ifndef BSZ
#define BSZ 256
#endif

#ifndef NLOOP
#define NLOOP 100
#endif

#ifndef FLOP
#define FLOP 0
#endif

static __global__ void kernel(State *state, size_t n, Real dt)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < n) {
    State s = state[i];

    for(int j = 0; j < NLOOP; ++j) {
      const State ds = rhs(s);

      #pragma unroll
      for(int k = 0; k < NVAR; ++k)
        ((Real *)&s)[k] += dt * ((Real *)&ds)[k]; // 2 * NVAR FLOP
    }

    state[i] = s;
  }
}

void evolve(void)
{
  using namespace global;

  cudaEventRecord(c0, 0);
  {
    const int bsz = BSZ;
    const int gsz = (n - 1) / bsz + 1;

    kernel<<<gsz, bsz>>>(s, n, 1.0e-3);
  }
  cudaEventRecord(c1, 0);
  cudaEventSynchronize(c1);

  float ns;
  cudaEventElapsedTime(&ns, c0, c1);
  ns /= NLOOP;

  std::cout
    << ns                                  << " ms/step, "
    << 1.0e-6 * n * (2 * NVAR + FLOP) / ns << " Gflops"
    << std::endl;
}
