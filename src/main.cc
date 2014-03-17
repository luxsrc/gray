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

#ifndef T_START
#define T_START 0
#endif

#ifndef N_DEFAULT
#define N_DEFAULT (512 * 512)
#endif

int main(int argc, char **argv)
{
  double t0 = T_START;
  size_t n  = N_DEFAULT;

  print("GRay: a massive parallel GRaysic integrator\n");
  debug("Debugging is turned on\n");

  Para para(argc, argv);
#ifdef ENABLE_GL
  vis::setup(argc, argv, para);
#endif

  Data data(n);
  data.init(t0);

#ifdef ENABLE_GL
  print("\
Press 'ESC' or 'q' to quit, 'p' to pulse, 'r' to reverse the run, 's' to\n\
to turn sprites on and off, and 'f' to enter and exit full screen\n\
");
#else
  print("Press 'Ctrl C' to quit\n");
#endif

  return para.solve(data);
}
