// Copyright (C) 2012--2014 Chi-kwan Chan
// Copyright (C) 2012--2014 Steward Observatory
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

#include "gray.h"

cudaError_t Data::dtoh()
{
  debug("Data::dtoh()\n");

  bool need_unmap = !mapped; // save the mapped state because device()
                             // will change it
  cudaError_t err =
    cudaMemcpy(buf, device(), sizeof(State) * n, cudaMemcpyDeviceToHost);

  if(need_unmap)
    deactivate(); // deactivate only if resource was not mapped before
                  // calling dtoh()
  return err;
}

cudaError_t Data::htod()
{
  debug("Data::htod()\n");

  bool need_unmap = !mapped; // save the mapped state because device()
                             // will change it
  cudaError_t err =
    cudaMemcpy(device(), buf, sizeof(State) * n, cudaMemcpyHostToDevice);

  if(need_unmap)
    deactivate(); // deactivate only if resource was not mapped before
                  // calling dtoh()
  return err;
}
