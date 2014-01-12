// Copyright (C) 2012,2013 Chi-kwan Chan
// Copyright (C) 2012,2013 Steward Observatory
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
#include "harm.h"
#include <cstdlib>
#include <para.h>

#ifndef T_START
#define T_START 0
#endif

#ifndef DT_DUMP
#define DT_DUMP 1
#endif

#ifndef N_DEFAULT
#define N_DEFAULT (512 * 512)
#endif

#ifndef WIDTH
#define WIDTH 512
#endif

#ifndef HEIGHT
#define HEIGHT 512
#endif

namespace global {
  double t        = T_START;
  double dt_dump  = DT_DUMP;
  double dt_saved = 0;
}

static void cleanup()
{
  if(harm::field) cudaFree(harm::field);
  if(harm::coord) cudaFree(harm::coord);
}

int main(int argc, char **argv)
{
  const char *name = NULL;

  print("GRay: a massive parallel GRaysic integrator\n");
  debug("Debugging is turned on\n");

#ifndef DISABLE_GL
  glutInit(&argc, argv);
  glutInitWindowSize(WIDTH, HEIGHT);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutCreateWindow(argv[0]);
  if(GL_NO_ERROR != glGetError())
    error("main(): fail to initialize GLUT/OpenGL\n");
#endif

  size_t n = 0, device = 0;
  for(int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if(arg[1] != '=')
      error("Unknown flag ""%s""\n", arg);
    else {
      switch(arg[0]) {
      case 'I': device          = atoi(arg + 2); break;
      case 'N': n               = atoi(arg + 2); break;
      case 'T': global::t       = atof(arg + 2); break;
      case 'D': global::dt_dump = atof(arg + 2); break;
      case 'H': name            =      arg + 2 ; break;
      default :
        if(!init_config(arg) || !prob_config(arg))
          error("Unknown parameter ""%s""\n", arg);
        break;
      }
      print("Set parameter ""%s""\n", arg);
    }
  }

  if(device > 0) {
    int n_devices;
    cudaGetDeviceCount(&n_devices);

    if(n_devices < 1)
      error("No GPU is found on this machine\n");
    else
      print("%d GPU%s found\n", n_devices, n_devices == 1 ? " is" : "s are");

    if(n_devices <= device)
      error("%u is an invalid GPU id");
    else
      print("Run on GPU %u", device);

    cudaSetDevice(device);
  }

  if(name) {
    using namespace harm;
    coord = load_coord("usgdump2d");
    field = load_field(name);
    if(coord && field && !atexit(cleanup))
      print("Loaded harm data from \"%s\"\n", name);
    else {
      if(field) cudaFree(field);
      if(coord) cudaFree(coord);
      error("Fail to load harm data from \"%s\"", name);
    }
  }

  Data data(n ? n : N_DEFAULT);
  init(data);

#ifndef DISABLE_GL
  vis((GLuint)data, (size_t)data);
  print("\
Press 'ESC' or 'q' to quit, 'p' to pulse, 'r' to reverse the run, 's' to\n\
to turn sprites on and off, and 'f' to enter and exit full screen\n\
");
#else
  print("Press 'Ctrl C' to quit\n");
#endif

  return solve(data);
}
