// Copyright (C) 2014 Chi-kwan Chan
// Copyright (C) 2014 Steward Observatory
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
#include <cstring>

Para::Para()
{
  debug("Para::Para()\n");
  cudaError_t err;

  init(buf);

  err = sync(&buf);
  if(cudaSuccess != err)
    error("Para::Para(): fail to synchronize parameters [%s]\n",
          cudaGetErrorString(err));

  /*
  for(int i = 1; i < argc; ++i) {
    if(strchr(argv[i], '='))
      print("Set parameter ""%s""\n", argv[i]);
    else
      error("Invalid argument ""%s""\n", argv[i]);
  }

  int i = 1;
  if(argc > i && argv[i][0] == '-') // `./gray -2` use the second device
    pick(atoi(argv[i++] + 1));
  else
    pick(0);

  size_t n = 0;
  for(; i < argc; ++i) {
    const char *arg = argv[i];
    if(arg[1] != '=')
      error("Unknown flag ""%s""\n", arg);
    else {
      switch(arg[0]) {
      case 'N': n            = atoi(arg + 2); break;
      case 'T': para.t       = atof(arg + 2); break;
      case 'D': para.dt_dump = atof(arg + 2); break;
      case 'O': para.format  =      arg + 2 ; break;
      case 'H': name         =      arg + 2 ; break;
      default :
        if(!init_config(arg) || !prob_config(arg))
          error("Unknown parameter ""%s""\n", arg);
        break;
      }
      print("Set parameter ""%s""\n", arg);
    }
  }
  */
}
