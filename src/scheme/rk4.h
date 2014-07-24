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

static inline __device__ real integrate(State &y, real t, real dt)
{
  const State s = y; // create a constant copy

  const State k1 = rhs(y, t);
  // Begin compute dt
  dt = dt > 0 ? getdt(y, t, k1, dt) : -getdt(y, t, k1, -dt);
  if(0 == dt) return 0;
  const real dt_2 = dt / 2;
  const real dt_6 = dt / 6;
  // End computing dt
  #pragma unroll
  EACH(y) = GET(s) + dt_2 * GET(k1);

  const State k2 = rhs(y, t + dt_2);
  #pragma unroll
  EACH(y) = GET(s) + dt_2 * GET(k2);

  const State k3 = rhs(y, t + dt_2);
  #pragma unroll
  EACH(y) = GET(s) + dt   * GET(k3);

  const State k4 = rhs(y, t + dt);
  #pragma unroll
  EACH(y) = GET(s) + dt_6 * (GET(k1) + 2 * (GET(k2) + GET(k3)) + GET(k4));

  return fixup(y, s, k1, dt);
}

double scheme::flop(void)
{
  return 4 + 12 * NVAR + 4 * FLOP_RHS + FLOP_GETDT;
}

double scheme::rwsz(void)
{
#if defined(RWSZ_RHS)
  return 2 * sizeof(State) + sizeof(size_t) + 4 * RWSZ_RHS;
#else
  return 2 * sizeof(State) + sizeof(size_t);
#endif
}
