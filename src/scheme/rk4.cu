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

static __device__ Var scheme(const Var v, const Real dt)
{
  const Real dt_2 = dt / 2;
  const Real dt_6 = dt / 6;
  const Real tmid = v.t + dt_2;
  const Real tend = v.t + dt;

  State y = v.s;

  const State k1 = rhs(y, v.t );
  #pragma unroll
  EACH(y) = GET(v.s) + dt_2 * GET(k1);

  const State k2 = rhs(y, tmid);
  #pragma unroll
  EACH(y) = GET(v.s) + dt_2 * GET(k2);

  const State k3 = rhs(y, tmid);
  #pragma unroll
  EACH(y) = GET(v.s) + dt   * GET(k3);

  const State k4 = rhs(y, tend);
  #pragma unroll
  EACH(y) = GET(v.s) + dt_6 * (GET(k1) + 2 * (GET(k2) + GET(k3)) + GET(k4));

  return (Var){y, tend};
}

static double flop(void)
{
  return 4 + 12 * NVAR + 4 * FLOP;
}
