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

Coord *harm::load_coord(Const &c, const char *name)
{
  FILE *file = fopen(name, "r");
  if(!file)
    error("ERROR: fail to open file \"%s\".\n", name);

  size_t count;
  fseek(file, 12, SEEK_CUR);
  if(fread(&c.n_r,     sizeof(size_t), 1, file) != 1 ||
     fread(&c.n_theta, sizeof(size_t), 1, file) != 1 ||
     fread(&c.n_phi,   sizeof(size_t), 1, file) != 1)
    error("ERROR: fail to read grid dimensions\n");
  c.n_rx = c.n_r - 1; // use power-law extrapolation for r > c.r[c.n_rx]
  count  = c.n_r * c.n_theta;

  double Gamma, a_spin = 0; // a_spin is uninitialized if Gamma cannot be read
  fseek(file, 56, SEEK_CUR);
  if(fread(&Gamma,    sizeof(double), 1, file) != 1 ||
     fread(&a_spin,   sizeof(double), 1, file) != 1)
    error("ERROR: fail to read Gamma or spin\n");
  fseek(file, 52, SEEK_CUR);
  c.Gamma  = Gamma;
  c.a_spin = a_spin;

  Coord *host;
  if(!(host = (Coord *)malloc(sizeof(Coord) * count)))
    error("ERROR: fail to allocate host memory\n");
  for(size_t i = 0; i < count; ++i) {
    double in[16];

    fseek(file, 4 + 3 * sizeof(size_t) + 3 * sizeof(double), SEEK_CUR);

    if(fread(in, sizeof(double), 2, file) != 2)
      error("ERROR: fail to read grid coordinates\n");
    if(i < c.n_r && i < N_R) c.r[i] = in[0];
    host[i].theta = in[1];

    fseek(file, 17 * sizeof(double), SEEK_CUR);

    if(fread(in, sizeof(double), 16, file) != 16)
      error("ERROR: fail to read metric\n");
    for(size_t j = 0; j < 16; ++j)
      (&(host[i].gcov[0][0]))[j] = in[j];

    fseek(file, 5 * sizeof(double), SEEK_CUR);

    if(fread(in, sizeof(double), 16, file) != 16)
      error("ERROR: fail to read dxdxp\n");
    for(size_t j = 0; j < 16; ++j)
      (&(host[i].dxdxp[0][0]))[j] = in[j];

    fseek(file, 52, SEEK_CUR);
  }
  fclose(file);

#if defined(N_IN) && defined(N_THETA)
  for(size_t i = 0; i < N_IN; ++i)
    for(size_t j = 0; j < c.n_theta; ++j)
      c.theta[j * N_IN + i] = host[j * c.n_r + i].theta;
#endif

  print("Data size = %zu x %zu x %zu\n"
        "Gamma = %g, spin parameter a = %g, rmin = %g, rmax = %g\n",
        c.n_r, c.n_theta, c.n_phi, c.Gamma, c.a_spin, c.r[0], c.r[c.n_r-1]);

  return host;
}
