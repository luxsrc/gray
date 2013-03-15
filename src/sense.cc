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

#include "gray.h"
#include <NiTE.h>

static bool first = true;

static void setup()
{
  print("Making sense...");

  if(nite::STATUS_OK != nite::NiTE::initialize())
    error("sense(): fail to initialize NiTE\n");

  first = false;

  print(" DONE\n");
}

static void cleanup()
{
  nite::NiTE::shutdown();
}

void sense()
{
  if(first && !atexit(cleanup)) setup();
}
