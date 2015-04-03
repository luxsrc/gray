# Copyright (C) 2015 Chi-kwan Chan
# Copyright (C) 2015 Steward Observatory
#
# This file is part of GRay.
#
# GRay is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GRay is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GRay.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import h5py  as h5
import os.path

def readline(file):
    """ Read a line from a binary file """
    line = "" # no prefix b because we want a string
    byte = file.read(1)
    while byte != b"\n": # compare to the byte b"\n"
        line += byte.decode("utf-8")
        byte  = file.read(1)
    return line

def isqrt(n):
    """ Integer square root """
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x

def load_raw(name):
    """ Load a GRay raw data file """
    with open(name, "rb") as file:
        print("Loading GRay raw file \"{0}\"".format(name))

        # Read the ASCII header
        head = np.array(list(map(float, readline(file).split())))
        n_nu = head.size // 2
        flux = head[0:n_nu]
        nu   = head[n_nu:2*n_nu]
        size = head[2*n_nu]

        # Read the binary images
        n    = isqrt(np.fromfile(file, dtype="q", count=1)[0])
        imgs = np.fromfile(file, dtype="f").reshape((n_nu, n, n))

        # Done
        return imgs, nu, size

def dump_hdf5(name, imgs, time, side, wavelength):
    """ Dump a GRay HDF5 file """
    if imgs.ndim != 3:
        raise NameError("imgs should be a 3 dimensional array")
    if imgs.shape[0] != len(time):
        raise NameError("The number of elements of time does not match "
                        "the zeroth dimension of imgs")

    n0 = imgs.shape[0]
    n1 = imgs.shape[1]
    n2 = imgs.shape[2]

    with h5.File(name, "w") as file: # FIXME: will file close automatically?
        print("Dumping GRay HDF5 file \"{0}\"".format(name))

        # Parameters
        file.attrs['units']      = "gcs"
        file.attrs['wavelength'] = wavelength
        # TODO: other simulation parameters

        # Create image array
        imgs = file.create_dataset("images", data=imgs,
                            chunks  =(1,    64, 64),
                            maxshape=(None, n1, n2))
        imgs.dims[0].label = "time"
        imgs.dims[1].label = "beta"
        imgs.dims[2].label = "alpha"

        # Create time series
        type   = np.dtype([('time',  "f"),
                           ('image', h5.special_dtype(ref=h5.Reference))])
        series = np.empty(n0, dtype=type)
        for i, t in enumerate(time):
            series[i] = (t, imgs.regionref[i,:,:])
        file.create_dataset("time_series", data=series)

        # Convert side into an array, compute physical scales
        side = side * ((np.arange(0, n2) + 0.5) / n1 - 0.5)
        G   = 6.67384e-8
        c   = 2.99792458e10
        t_g = G * 4.3e6 * 1.99e33 / (c * c * c) # ~ 21.2 s
        r_g = G * 4.3e6 * 1.99e33 / (c * c)     # ~ 6.35e11 cm

        # Attach scales in gravitational units
        dscl = file.create_group("dimension scales/gravitational units")
        time = dscl.create_dataset("time", data=time, maxshape=None)
        side = dscl.create_dataset("side", data=side)

        imgs.dims.create_scale(time, "GM/c^3")
        imgs.dims.create_scale(side, "GM/c^2")

        imgs.dims[0].attach_scale(time)
        imgs.dims[1].attach_scale(side)
        imgs.dims[2].attach_scale(side)

        # Attach scales in physical (cgs) units
        dscl = file.create_group("dimension scales/physical units (cgs)")
        time = dscl.create_dataset("time", data=time[:] * t_g, maxshape=None)
        side = dscl.create_dataset("side", data=side[:] * r_g)

        imgs.dims.create_scale(time, "s")
        imgs.dims.create_scale(side, "cm")

        imgs.dims[0].attach_scale(time)
        imgs.dims[1].attach_scale(side)
        imgs.dims[2].attach_scale(side)

def load(name):
    ext = os.path.splitext(name)[1][1:]
    if ext == "raw":
        return load_raw(name)
    else:
        raise NameError("Fail to load file \"{0}\", "
                        "which is in an unsupported format".format(name))

def dump(name, imgs, nu, side, time):
    ext = os.path.splitext(name)[1][1:]
    if ext == "h5" or ext == "hdf5":
        c = 2.99792458e10
        dump_hdf5(name, imgs, time, side, nu / c)
    else:
        raise NameError("Fail to dump file \"{0}\", "
                        "which is in an unsupported format".format(name))
