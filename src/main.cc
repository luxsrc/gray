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

#include "geode.h"

int main(int argc, char **argv)
{
  std::cout
    << "Geode: a massive parallel geodesic integrator"
    << std::endl;

  setup(argc, argv);

  std::cout <<
#ifndef DISABLE_GL
"Press 'ESC' or 'q' to quit, 'p' to pulse, 'r' to reverse the run, and 's'\n"
"to turn sprites on and off"
#else
"Press 'Ctrl C' to quit"
#endif
    << std::endl;

  return solve();
}
