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
#include <cstdlib>
#include <cmath>

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
  for(int i = 0; i < N_NU; ++i)
    c.nu0[i] = pow(10.0, 9 + 0.5 * i);

  c.coord = NULL;
  c.field = NULL;
}

const char *Para::config(Const &c, const char *arg)
{
  const char *val;

       if((val = match("i",  arg))) c.i_obs   = atof(val);
  else if((val = match("a",  arg))) c.a_spin  = atof(val);
  else if((val = match("ne", arg))) c.ne_rho  = atof(val);
  else if((val = match("rd", arg))) c.Tp_Te_d = atof(val);
  else if((val = match("rw", arg))) c.Tp_Te_w = atof(val);
  // TODO: load nu0 and HARM data

  return val;
}
