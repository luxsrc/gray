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

#include "geode.h"

cudaError_t Data::d2h()
{
  debug("Data::d2h()\n");

  bool need_unmap = !mapped; // save the mapped state because device()
                             // will change it
  cudaError_t err =
    cudaMemcpy(buf, device(), sizeof(State) * n, cudaMemcpyDeviceToHost);

  if(need_unmap)
    deactivate(); // deactivate only if resource was not mapped before
                  // calling d2h()
  return err;
}

cudaError_t Data::h2d()
{
  debug("Data::h2d()\n");

  bool need_unmap = !mapped; // save the mapped state because device()
                             // will change it
  cudaError_t err =
    cudaMemcpy(device(), buf, sizeof(State) * n, cudaMemcpyHostToDevice);

  if(need_unmap)
    deactivate(); // deactivate only if resource was not mapped before
                  // calling d2h()
  return err;
}
