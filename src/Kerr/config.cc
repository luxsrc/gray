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

void Para::init(Const &c)
{
  c.r_obs     = 1000;
  c.i_obs     = 30;
  c.a_spin    = 0.999;
  c.dt_scale  = 1.0 / 32;
  c.epsilon   = 1e-3;
  c.tolerance = 1e-1;

  c.coord = NULL;
  c.field = NULL;

  c.nr = c.ntheta = c.nphi = 0;
  c.lnrmin = c.lnrmax = 0;

  c.m_BH    = 4.3e6; // in unit of solar mass
  c.Gamma   = 4.0 / 3;
  c.Tp_Te_d = 3;
  c.Tp_Te_w = 3;
  c.T_w     = 0;
  c.ne_rho  = 1e6;
  for(int i = 0; i < N_NU; ++i)
    c.nu0[i] = 0;
}
