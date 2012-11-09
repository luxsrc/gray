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

static void idle(void)
{
  using namespace global;

  cudaEventRecord(c0, 0);
  evolve(); // n * 21 FLOP
  cudaEventRecord(c1, 0);
  cudaEventSynchronize(c1);

  float ns;
  cudaEventElapsedTime(&ns, c0, c1);

  std::cout
    << ns                   << " ms/step, "
    << 1.0e-6 * n * 21 / ns << " Gflops"
    << std::endl;

  vis();

  glutPostRedisplay();
}

int solve(void)
{
  vis();

  glutIdleFunc(idle);
  glutMainLoop();

  return 0;
}
