// Copyright (C) 2013,2014 Chi-kwan Chan
// Copyright (C) 2013,2014 Steward Observatory
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
#include <cstdlib>
#include <cmath>

Field *harm::load_field(Const &c, const char *name)
{
  FILE *file = fopen(name, "r");
  if(!file)
    error("ERROR: fail to open file \"%s\".\n", name);

  double time;
  size_t n1, n2, n3, count;
  if(fscanf(file, "%lf %zu %zu %zu", &time, &n1, &n2, &n3) != 4)
    error("ERROR: fail to read time or grid size\n");
  while('\n' != fgetc(file));
  count = c.n_r * c.n_theta * c.n_phi;
  if(n1 != c.n_r || n2 != c.n_theta || n3 != c.n_phi)
    error("ERROR: inconsistent grid size\n");

  Field *host;
  if(!(host = (Field *)malloc(sizeof(Field) * count)))
    error("ERROR: fail to allocate host memory\n");
  if(fread(host, sizeof(Field), count, file) != count)
    error("ERROR: fail to read data\n");
  fclose(file);

  real dmax = 0, Emax = 0, vmax = 0, Bmax = 0, Tmax = 0;
  for(size_t i = 0; i < count; ++i) {
    const real v = sqrt(host[i].v1 * host[i].v1 +
                        host[i].v2 * host[i].v2 +
                        host[i].v3 * host[i].v3);
    const real B = sqrt(host[i].B1 * host[i].B1 +
                        host[i].B2 * host[i].B2 +
                        host[i].B3 * host[i].B3);
    const real T = host[i].u / host[i].rho;

    if(dmax < host[i].rho) dmax = host[i].rho;
    if(Emax < host[i].u  ) Emax = host[i].u;
    if(vmax < v          ) vmax = v;
    if(Bmax < B          ) Bmax = B;
    if(Tmax < T          ) Tmax = T;
  }

  print("Maximum density        = %g\n"
        "Maximum energy         = %g\n"
        "Maximum temperature    = %g\n"
        "Maximum speed          = %g\n"
        "Maximum magnetic field = %g\n",
        dmax, Emax, Tmax, vmax, Bmax);

  return host;
}
