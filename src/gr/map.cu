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

static __device__ Point map(State s)
{
  if(s.r <= 1 + sqrt(1 - A_SPIN * A_SPIN))
    return (Point){0, 0, 0, 0, 0, 0};

  const Real R = s.r * sin(s.theta);
  const Real x = R   * cos(s.phi  );
  const Real y = R   * sin(s.phi  );
  const Real z = s.r * cos(s.theta);

  return (Point){x, y, z, s.kr, fabs(s.ktheta), fabs(s.bimpact)};
}
