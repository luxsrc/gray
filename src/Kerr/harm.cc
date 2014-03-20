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

#include "../gray.h"
#include <cstring>

bool harm::load(Const &c, const char *dump)
{
  char grid[1024], *p;
  strcpy(grid, dump);
  p = grid + strlen(grid);
  while(p > grid && *p != '/') --p;
  strcpy(*p == '/' ? p + 1 : p, "usgdump2d");

  c.coord = harm::load_coord(c, grid);
  c.field = harm::load_field(c, dump);

  if(c.coord && c.field) {
    print("\
Loaded harm grid from \"%s\"\n\
        and data from \"%s\"\n\
", grid, dump);
    return true;
  } else {
    print("\
Failed to load harm grid from \"%s\"\n\
                 or data from \"%s\"\n\
", grid, dump);
    return false;
  }
}
