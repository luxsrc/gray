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

#include "gray.h"

void optimize(int device)
{
  int n_devices;
  cudaGetDeviceCount(&n_devices);

  if(n_devices < 1)
    error("No GPU is found on this machine\n");
  else
    print("%d GPU%s found\n", n_devices, n_devices == 1 ? " is" : "s are");

  if(n_devices <= device)
    error("%u is an invalid GPU id\n");
  else
    print("Run on GPU %u\n", device);

  cudaError_t err = cudaSetDevice(device);
  if(cudaSuccess != err)
    error("init(): fail to switch to device %d [%s]\n",
          device, cudaGetErrorString(err));
}
