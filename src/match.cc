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
#include <cstring>

const char *match(const char *sym, const char *arg)
{
  const size_t l = strlen(sym);
  const size_t n = strlen(arg);

  if(l + 1 > n || '=' != arg[l])
    return NULL;

  for(size_t i = 0; i < l; ++i)
    if(sym[i] != arg[i])
      return NULL;

  return arg + l + 1;
}
