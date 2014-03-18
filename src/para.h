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

#ifndef PARA_H
#define PARA_H

class Para {
  Const buf; // host buffer, device resource is defined as constant memory

  void        init(Const &); // implemented in "src/*/config.cc"
  cudaError_t sync(Const *); // implemented in "src/core.cu"

 public:
  Para();
  ~Para();
};

#endif // PARA_H
