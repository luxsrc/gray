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

Para::Para()
{
  debug("Para::Para()\n");
  cudaError_t err;

  define(buf);

  if(cudaSuccess != (err = sync(&buf)))
    error("Para::Para(): fail to synchronize parameters [%s]\n",
          cudaGetErrorString(err));
}

Para::~Para()
{
  debug("Para::~Para()\n");
}

bool Para::config(const char *arg)
{
  debug("Para::config(\"%s\")\n", arg);
  cudaError_t err;

  const char *val = config(buf, arg);

  if(cudaSuccess != (err = sync(&buf)))
    error("Para::Para(): fail to synchronize parameters [%s]\n",
          cudaGetErrorString(err));

  return NULL != val;
}
