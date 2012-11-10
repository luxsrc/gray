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
  const float rr =  s.x * s.x + s.y * s.y + s.z * s.z;
  const float vr = (s.x * s.u + s.y * s.v + s.z * s.w) / rr;

  const float u  = s.u - s.x * vr;
  const float v  = s.v - s.y * vr;
  const float w  = s.w - s.z * vr;

  const float lx = s.y * s.w - s.z * s.v;
  const float ly = s.z * s.u - s.x * s.w;
  const float lz = s.x * s.v - s.y * s.u;

  const float E  = (s.u * s.u + s.v * s.v + s.w * s.w) / 2;
  const float El = (u * u + v * v + w * w) / 2;
  const float Er = E - El;
  const float l  = sqrt(lx * lx + ly * ly + lz * lz);

  return (Point){s.x, s.y, s.z, E * El * 10, E * l / 5, E * Er * 10};
}
