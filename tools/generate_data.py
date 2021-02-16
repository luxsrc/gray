#!/usr/bin/env python3

import h5py
import numpy as np

times = ["-0", "-10", "-20", "-40"]
Nx = 20
Ny = 20
Nz = 20
xx = np.linspace(-300, 300, Nx, dtype=np.single)
yy = np.linspace(-300, 300, Ny, dtype=np.single)
zz = np.linspace(-300, 300, Nz, dtype=np.single)

zeros = np.zeros(Nx * Ny * Nz, dtype=np.single)
ones = np.ones(Nx * Ny * Nz, dtype=np.single)

dims = ["t", "x", "y", "z"]

with h5py.File("data.h5", "w") as f:
    grid_group = f.create_group("grid")
    grid_group.create_dataset("x", data=xx)
    grid_group.create_dataset("y", data=yy)
    grid_group.create_dataset("z", data=zz)
    for time in times:
        it_group = f.create_group(time)
        for i in range(len(dims)):
            for j in range(len(dims)):
                for k in range(j, len(dims)):
                    name = f"Gamma_{dims[i]}{dims[j]}{dims[k]}"
                    it_group.create_dataset(
                        name,
                        # data=np.random.randn(Nx * Ny * Nz).astype('f'),
                        data=ones,
                    )
