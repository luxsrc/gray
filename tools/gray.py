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
        nus  = head[n_nu:2*n_nu]
        size = head[2*n_nu]

        # Read the binary images
        n    = isqrt(np.fromfile(file, dtype="q", count=1)[0])
        imgs = np.fromfile(file, dtype="f").reshape((n_nu, n, n))

        # Done
        return imgs, nus, size * ((np.arange(0, n) + 0.5) / n - 0.5)
