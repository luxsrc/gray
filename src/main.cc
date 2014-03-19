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
#include <cstdlib>

#ifndef N_DEFAULT
#define N_DEFAULT (512 * 512)
#endif

#ifndef T_START
#define T_START 0
#endif

#ifndef DT_DUMP
#define DT_DUMP 1
#endif

int main(int argc, char **argv)
{
  int    gpu = 0;
  size_t n   = N_DEFAULT;
  double t0  = T_START;
  double dt  = DT_DUMP;

  const char *format = NULL; // no snapshot by default
  const char *name   = "out.raw";

  print("GRay: a massive parallel ODE integrator written in CUDA C/C++\n");
#ifdef ENABLE_GL
  print("Press 'q' to quit, 'p' and 'r' to pulse and reverse the run\n");
#else
  print("Press 'Ctrl C' to quit\n");
#endif
  debug("Debugging is turned on\n");

  Para para;
  for(int i = 1; i < argc; ++i) {
    const char *arg = argv[i], *val;

         if((val = match("gpu",      arg))) gpu    = atoi(val);
    else if((val = match("n",        arg))) n      = atoi(val);
    else if((val = match("t0",       arg))) t0     = atof(val);
    else if((val = match("dt",       arg))) dt     = atof(val);
    else if((val = match("snapshot", arg))) format =      val ;
    else if((val = match("output",   arg))) name   =      val ;

    if(val || para.config(arg))
      print("Set parameter \"%s\"\n", arg);
    else
      error("Unknown argument \"%s\"\n", arg); // it's wasteful to run the
                                               // wrong simulation
  }
  pick(gpu); // TODO: print GPU info from main() instead of pick()?

  Data data(n);
  data.init(t0);
  data.snapshot(format);

  float ms, actual, peak;
  while(size_t c = data.solve(dt, ms, actual, peak)) {
    print("t = %.2f; %.0f ms/%.0f steps ~ %.2f Gflops (%.2f%%), %.2f GB/s\n",
          data.t, ms, actual,
          1e-6 * scheme::flop() * actual / ms, 100 * actual / peak,
          1e-6 * (24 * sizeof(real) * actual + scheme::rwsz() * n) / ms);
    data.snapshot(c == LIMIT ? format : NULL);
  }

  data.output(name);
  return 0;
}
