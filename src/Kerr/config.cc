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

#include "../gray.h"
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>

static size_t fill(real *nu, const char *val)
{
  size_t n = 0;

  if(FILE *file = fopen(val, "r")) { // a file is provided
    double tmp;
    while(n < N_NU && fscanf(file, "%lf", &tmp) > 0)
      nu[n++] = tmp;
    fclose(file);
  } else { // hopefully a list of number
    char *src = (char *)malloc(strlen(val) + 1);
    strcpy(src, val);
    char *tok = strtok(src, " \t\n,;");
    while(n < N_NU && tok) {
      nu[n++] = atof(tok);
      tok = strtok(NULL, " \t\n,;");
    }
    free(src);
  }

  if(n) {
    print("%zu freq: ", n);
    for(int i = 0; i < n-1; print("%g,", nu[i++]));
    print("%g\n", nu[n-1]);
  }

  return n;
}

static bool load(Const &c, const char *dump)
{
  if(c.coord) cudaFree(c.coord);
  if(c.field) cudaFree(c.field);

  char grid[1024], *p;
  strcpy(grid, dump);
  p = grid + strlen(grid);
  while(p > grid && *p != '/') --p;
  strcpy(*p == '/' ? p + 1 : p, "usgdump2d");

  c.coord = harm::load_coord(c, grid);
  c.field = harm::load_field(c, dump);

  if(c.coord && c.field) {
    print("\
Loaded harm grid from \"%s\"\n\
        and data from \"%s\"\n\
", grid, dump);
    return true;
  } else {
    print("\
Failed to load harm grid from \"%s\"\n\
                 or data from \"%s\"\n\
", grid, dump);
    return false;
  }
}

void Para::define(Const &c)
{
  c.r_obs     = 1000;
  c.i_obs     = 30;
  c.a_spin    = 0.999;
  c.dt_scale  = 1.0 / 32;
  c.epsilon   = 1e-3;
  c.tolerance = 1e-1;

  c.m_BH      = 4.3e6; // in unit of solar mass
  c.ne_rho    = 1e6;
  c.threshold = 5;
  c.Tp_Te_d   = 3;
  c.Tp_Te_w   = 3;
  c.n_nu      = 0;

  c.coord = NULL;
  c.field = NULL;
}

bool Para::config(Const &c, const char *arg)
{
  const char *val;

       if((val = match("i",    arg))) c.i_obs   = atof(val);
  else if((val = match("a",    arg))) c.a_spin  = atof(val);
  else if((val = match("ne",   arg))) c.ne_rho  = atof(val);
  else if((val = match("rd",   arg))) c.Tp_Te_d = atof(val);
  else if((val = match("rw",   arg))) c.Tp_Te_w = atof(val);
  else if((val = match("nu",   arg))) return 0 < (c.n_nu = fill(c.nu0, val));
  else if((val = match("harm", arg))) return load(c, val);

  return NULL != val;
}
