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
  size_t n = 0;
  State *s = NULL;
}

static void cleanup(void)
{
  delete[] global::s;
}

int main(int argc, char **argv)
{
  std::cout
    << "Geode: a massive parallel geodesic integrator"
    << std::endl;

  global::n = 1024;
  global::s = new State[global::n];

  for(size_t i = 0; i < global::n; ++i) {
    global::s[i].x = 20.0 * ((double)rand() / RAND_MAX - 0.5);
    global::s[i].y = 20.0 * ((double)rand() / RAND_MAX - 0.5);
    global::s[i].z = 20.0 * ((double)rand() / RAND_MAX - 0.5);
    global::s[i].u =  0.5 * ((double)rand() / RAND_MAX + 1.0);
    global::s[i].v =  0.5 * ((double)rand() / RAND_MAX + 1.0);
    global::s[i].w =  0.5 * ((double)rand() / RAND_MAX + 1.0);
  }

  if(!atexit(cleanup)) setup(argc, argv);

  std::cout
    << "Press 'ESC' or 'q' to quit"
    << std::endl;

  return solve();
}
