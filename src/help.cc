// Copyright (C) 2015 Chi-kwan Chan
// Copyright (C) 2015 Steward Observatory
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

int help(const char *argv0)
{
  print("usage: %s [OPTION] [PARAMETER=VALUE ...]\n", argv0);
  print("\n\
Available OPTION includes:\n\
     --help  display this help and exit\n\
\n\
Available PARAMETER includes:\n\
     gpu    gpu id for running the job\n\
     n      total number of rays\n\
     t0     start time\n\
     dt     separation time between each snapshot\n\
     imgs   name of the output file that stores images\n\
     rays   name of the output file that stores full rays\n\
     grid   name of the output file that stores the source grid\n\
\n\
     imgsz  size of the computed image in GM/c^2\n\
     imgx0  horizontal coordinate of the image center\n\
     imgy0  vertical coordinate of the image center \n\
     r      distance between the image plane and the black hole\n\
     i      inclination angle in degrees\n\
     j      azimuthal angle in degrees\n\
     ne     normalization for the electron number density\n\
     beta   threshold plasma beta to distinguish the disk and funnel\n\
     td     electron-ion temperature ratio for the disk\n\
     tf     electron-ion temperature ratio for the funnel\n\
     Tf     constant temperature for the funnel\n\
     harm   snapshots of GRMHD simulations created by HARM\n\
     nu     frequencies in Hz; can be a list of numbers or a file name\n\
\n");
  print("Report bugs to <chanchikwan@gmail.com>.\n");

  return 0;
}
