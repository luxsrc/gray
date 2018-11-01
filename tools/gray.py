# Copyright (C) 2015,2016 Chi-kwan Chan & Lia Medeiros
# Copyright (C) 2015,2016 Steward Observatory
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

from os import path
import numpy as np

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

def load_imgs(name):
    """ Load a GRay raw image(s) file """
    ext = path.splitext(name)[1][1:]
    if ext != "imgs" and ext != "raw":
        raise NameError("Fail to load file \"{0}\", "
                        "which is in an unsupported format".format(name))

    with open(name, "rb") as file:
        print("Loading GRay raw image(s) file \"{0}\"".format(name))

        # Read the ASCII header
        head = np.array(list(map(float, readline(file).split())))
        n_nu = head.size // 2
        nus  = head[n_nu:2*n_nu]
        size = head[2*n_nu]

        # Read the binary images
        n    = isqrt(np.fromfile(file, dtype="q", count=1)[0])
        imgs = np.fromfile(file, dtype="f").reshape((n_nu, n, n))

        # Done
        return imgs, nus, size * ((np.arange(0, n) + 0.5) / n - 0.5)

def load_rays(name):
    """ Load a GRay ray(s) file """
    ext = path.splitext(name)[1][1:]
    if ext != "rays":
        raise NameError("Fail to load file \"{0}\", "
                        "which is in an unsupported format".format(name))

    r = []
    with open(name, "rb") as f:
        # Number of rays and number of variables for each point in a ray
        n = np.fromfile(f, np.uint64, 2)

        # For each ray...
        for i in range(n[0]):
            c = np.fromfile(f, np.uint64,  1) # number of points in a ray
            d = np.fromfile(f, np.float32, c[0] * n[1])
            r.append(d.reshape(c[0], n[1]))

    # Done
    return r

def d(f):
    return np.hstack((f[1]-f[0], (f[2:]-f[:-2])/2, f[-1]-f[-2]))
