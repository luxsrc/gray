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

def dump_hdf5(name, imgs, time, side, parameters):
    """ Dump GRay data into a new HDF5 file """
    with h5.File(name, "w") as file: # FIXME: will file close automatically?
        print("Dumping GRay HDF5 file \"{0}\"".format(name))

        # Turn parameters into file attributes
        for key, value in parameters.items():
            file.attrs[key] = value

        # Create image array/dataset
        maxs = (None, imgs.shape[1], imgs.shape[2])
        imgs = file.create_dataset("images", data=imgs,
                                   maxshape=maxs, chunks=(1, 64, 64))
        imgs.dims[0].label = "time"
        imgs.dims[1].label = "beta"
        imgs.dims[2].label = "alpha"

        # Create dimension scales
        time = file.create_dataset("time", data=time,
                                   maxshape=(None,), chunks=True)
        side = file.create_dataset("side", data=side)
        imgs.dims.create_scale(time)
        imgs.dims.create_scale(side)

        # Attach dimension scales to image array/dataset
        imgs.dims[0].attach_scale(time)
        imgs.dims[1].attach_scale(side)
        imgs.dims[2].attach_scale(side)

def append_hdf5(name, imgs, time):
    """ Dump GRay data into an existing HDF5 file """
    with h5.File(name, "r+") as file: # FIXME: weill file close automatically?
        print("Appending GRay HDF5 file \"{0}\"".format(name))

        nt = file['images'].shape[0]

        file['images'].resize(nt + imgs.shape[0], axis=0)
        file['images'][nt:,:,:] = imgs

        file['time'].resize(nt + imgs.shape[0], axis=0)
        file['time'][nt:] = time

def load(name):
    ext = os.path.splitext(name)[1][1:]
    if ext == "raw":
        return load_raw(name)
    else:
        raise NameError("Fail to load file \"{0}\", "
                        "which is in an unsupported format".format(name))

def dump(name, imgs, time, nu=None, side=None):
    if imgs.ndim != 3 or time.ndim != 1:
        raise NameError("Unexpected number of dimensions")
    if imgs.shape[1] != imgs.shape[2]:
        raise NameError("The images are not square")
    if imgs.shape[0] != time.shape[0]:
        raise NameError("The number of elements of time does not match "
                        "the zeroth dimension of imgs")

    ext = os.path.splitext(name)[1][1:]
    if ext == "h5" or ext == "hdf5":
        if os.path.isfile(name) and nu == None and side == None:
            append_hdf5(name, imgs, time)
        elif nu != None and side != None:
            ns = imgs.shape[1]
            dump_hdf5(name, imgs, time,
                      side * ((np.arange(0, ns) + 0.5) / ns - 0.5),
                      {'wavelength (cm)': nu / 2.99792458e10})
        else:
            raise NameError("Variables nu and side are required "
                            "only for creating dump new HDF5 file")
    else:
        raise NameError("Fail to dump file \"{0}\", "
                        "which is in an unsupported format".format(name))
