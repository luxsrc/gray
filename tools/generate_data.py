#!/usr/bin/env python3

# Copyright (C) 2020-2021 Gabriele Bozzola
#
# This program is free software; you can redistribute it and/or modify it under the terms
# of the GNU General Public License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this
# program; if not, see <https://www.gnu.org/licenses/>.

import concurrent.futures

import h5py
import numpy as np

from boosted_ks import Christoffel, metric

"""Prepare a HDF5 file in the format required by GRay2 containing spacetime and fluid
information. The output file will have multiple HDF5 groups, one called 'grid' contains
the coordinates, the other ones have as name the timestep at which the variables are
defined. Each of these groups contain numerous datasets, one for each variable."""

# User controllable parameters

# KS parameters
a_spin = 0.6
boostv = 0.5

# File parameters
output_file = "data.h5"
times = ["0"]
precision = np.single
num_points_x, num_points_y, num_points_z = 10, 12, 14
xmin, xmax = -300, 300
ymin, ymax = -400, 400
zmin, zmax = -600, 600
# If num_worker is not None, use this many CPUs for the following computations. If it is
# None, use as many as possible.
num_workers = None


_dims = ("t", "x", "y", "z")

def fisheye(coord, min_, max_, num_points):
    """Fisheye transformation.

    Takes in the logically-Cartesian coordinates, and returns the corresponding fisheye
    coordinates as defined from min_ to max_.

    """
    B = np.cbrt(np.arcsinh(min_))
    A = (np.cbrt(np.arcsinh(max_)) - B) / (num_points - 1)
    return np.sinh((A * coord + B) ** 3)


# Values 0, 1, 2, 3, .... num_points - 1
cart_x = np.linspace(0, num_points_x - 1, num_points_x, dtype=precision)
cart_y = np.linspace(0, num_points_y - 1, num_points_y, dtype=precision)
cart_z = np.linspace(0, num_points_z - 1, num_points_z, dtype=precision)

# Physical coordinates
xx = fisheye(cart_x, xmin, xmax, num_points_x)
yy = fisheye(cart_y, ymin, ymax, num_points_y)
zz = fisheye(cart_z, zmin, zmax, num_points_z)

# Now we have to prepare all the variables.

# This can be computationally expensive, so we are going to distribute the computation on
# as many workers as we can.

# Gamma dict is a dictionary with keys the times and values another dictionary that has
# as keys the indices and as values the Christoffel symbols. Similarly, metric dict. The
# other dicts have only one level.
Gamma_dict = {}
metric_dict = {}

fluid_vars = ['rho']
# The fluid variables have to follow this naming convention in this file. They have to
# be called name_dict, where name is one of those that enter fluid_vars.
rho_dict = {}

# Not very Pythonic
indices = []
indices_metric = []
for i in range(4):
    for j in range(4):
        for k in range(j, 4):
            indices += [(i, j, k)]
            if i == 0:
                indices_metric += [(j, k)]

for time in times:
    print(f"Working on time {time}")

    def compute_Gamma(ind):
        """Compute the Christoffel symbol with given indices."""
        return [[[Christoffel((1, a_spin, boostv), (float(time), x, y, z), *ind)
                  for x in xx] for y in yy] for z in zz]

    def compute_metric(ind):
        """Compute the metric component with given indices."""
        return [[[metric((1, a_spin, boostv), (float(time), x, y, z), *ind)
                  for x in xx] for y in yy] for z in zz]

    Gamma_dict[time] = {}
    metric_dict[time] = {}

    # Do the actual computation
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as exe:
        for index, Gamma in zip(indices, exe.map(compute_Gamma, indices)):
            Gamma_dict[time][index] = Gamma
    print("Computed Gammas")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as exe:
        for index, met in zip(indices_metric, exe.map(compute_metric, indices_metric)):
            metric_dict[time][index] = met
    print("Computed metric")

    def compute_rho(x, y, z):
        """Compute the density as 1/r."""
        return 1/np.sqrt(x*x + y*y + z*z)

    rho_dict[time] = [[[compute_rho(x, y, z) for x in xx] for y in yy] for z in zz]
    print("Computed rho")


with h5py.File(output_file, "w") as f:
    grid_group = f.create_group("grid")
    grid_group.create_dataset("x", data=xx)
    grid_group.create_dataset("y", data=yy)
    grid_group.create_dataset("z", data=zz)
    for time in times:
        it_group = f.create_group(time)
        for ind in indices:
            i, j, k = ind
            name = f"Gamma_{_dims[i]}{_dims[j]}{_dims[k]}"
            data = Gamma_dict[time][(i, j, k)]
            data = np.nan_to_num(data)
            data = data.astype(precision)
            it_group.create_dataset(name, data=data)
        for ind in indices_metric:
            i, j = ind
            name = f"g_{_dims[i]}{_dims[j]}"
            data = metric_dict[time][(i, j)]
            data = np.nan_to_num(data)
            data = data.astype(precision)
            it_group.create_dataset(name, data=data)
        # Fluid variables
        for var in fluid_vars:
            name = var
            # We read the _dict variables from the global namespace
            data = globals()[f"{var}_dict"][time]
            data = np.nan_to_num(data)
            data = data.astype(precision)
            it_group.create_dataset(name, data=data)
