#!/usr/bin/env python3

import h5py
import numpy as np

time = str(0.08)
Nx = 120
Ny = 120
Nz = 120
xx = np.linspace(-300, 300, Nx, dtype=np.single)
yy = np.linspace(-300, 300, Ny, dtype=np.single)
zz = np.linspace(-300, 300, Nz, dtype=np.single)

zeros = np.zeros(Nx * Ny * Nz, dtype=np.single)
ones = np.ones(Nx * Ny * Nz, dtype=np.single)

dims = ["t", "x", "y", "z"]

with h5py.File("data.h5", "w") as f:
    it_group = f.create_group(time)
    it_group.create_dataset("x", data=xx)
    it_group.create_dataset("y", data=yy)
    it_group.create_dataset("z", data=zz)
    for i in range(len(dims)):
        for j in range(len(dims)):
            for k in range(j, len(dims)):
                name = f"Gamma_{dims[i]}{dims[j]}{dims[k]}"
                it_group.create_dataset(name, data=zeros)
