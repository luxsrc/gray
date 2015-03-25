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
    for(size_t i = 0; i < n-1; print("%g,", nu[i++]));
    print("%g\n", nu[n-1]);
  }

  return harm::n_nu = n;
}

void Para::define(Const &c)
{
  c.imgsz     = 64;
#ifdef ENABLE_GL
  c.r_obs     = 256;
#else
  c.r_obs     = 1024;
#endif
  c.i_obs     = 60;
  c.j_obs     = 90;
  c.a_spin    = 0.999;
  c.dt_scale  = 1.0 / 32;
#ifdef ENABLE_GL
  c.epsilon   = 1.0 / 32; // do get too close so we can get out...
#else
  c.epsilon   = 1e-3;
#endif
  c.tolerance = 1e-1;

  c.m_BH      = 4.3e6; // in unit of solar mass
  c.ne_rho    = 1e6;
  c.threshold = 5; // = 1 / beta_threshold
  c.tgas_max  = 1;
  c.Ti_Te_d   = 3;
  c.Ti_Te_f   = 3;
  c.n_nu      = 0;

  c.coord = NULL;
  c.field = NULL;
}

bool Para::config(Const &c, const char *arg)
{
  const char *val;

       if((val = match("imgsz", arg))) c.imgsz     =   atof(val);
  else if((val = match("r",     arg))) c.r_obs     =   atof(val);
  else if((val = match("i",     arg))) c.i_obs     =   atof(val);
  else if((val = match("j",     arg))) c.j_obs     =   atof(val);
  else if((val = match("a",     arg))) c.a_spin    =   atof(val);
  else if((val = match("ne",    arg))) c.ne_rho    =   atof(val);
  else if((val = match("beta",  arg))) c.threshold = 1/atof(val);
  else if((val = match("td",    arg))) c.Ti_Te_d   = 1/atof(val);
  else if((val = match("tf",    arg))) c.Ti_Te_f   = 1/atof(val);
  else if((val = match("Tf",    arg))) c.Ti_Te_f   = - atof(val);
  else if((val = match("rx",    arg))) return harm::setx(c, val);
  else if((val = match("harm",  arg))) return harm::load(c, val);
  else if((val = match("nu",    arg))) return 0 < (c.n_nu = fill(c.nu0, val));

  return NULL != val;
}
