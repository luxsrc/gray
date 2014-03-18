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

#ifndef DT_DUMP
#define DT_DUMP 1
#endif

#ifndef N_DEFAULT
#define N_DEFAULT (512 * 512)
#endif

namespace global {
  double dt_dump = DT_DUMP;
}

int main(int argc, char **argv)
{
  double t0 = T_START;
  size_t n  = N_DEFAULT;

  const char *format = "%03d.raw";
  const char *output = "out.raw";

  print("GRay: a massive parallel GRaysic integrator\n");
  debug("Debugging is turned on\n");

#ifdef ENABLE_GL
  print("\
Press 'ESC' or 'q' to quit, 'p' to pulse, 'r' to reverse the run, 's' to\n\
to turn sprites on and off, and 'f' to enter and exit full screen\n\
");
#else
  print("Press 'Ctrl C' to quit\n");
#endif

  Para para;
  /* TODO: configure parameters and change n, t0, etc
  for(int i = 1; i < argc; ++i)
    para.config(argv[i]);
  */
  pick(0);

  Data data(n);
  data.init(t0);
  data.dump(format);

  float ms, actual, peak;
  while(size_t c = data.solve(global::dt_dump, ms, actual, peak)) {
    print("t = %.2f; %.0fms/%.0fsteps ~ %.2f Gflops (%.2f%%), %.2fGB/s\n",
          data.t, ms, actual,
          1e-6 * scheme::flop() * actual / ms, 100 * actual / peak,
          1e-6 * (24 * sizeof(real) * actual + scheme::rwsz() * n) / ms);
    data.dump(c == LIMIT ? format : NULL);
  }

  data.spec(output);
  return 0;
}
