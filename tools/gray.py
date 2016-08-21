# Copyright (C) 2016 Chi-kwan Chan
# Copyright (C) 2016 Steward Observatory
#
# This file is part of GRay2.
#
# GRay2 is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GRay2 is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GRay2.  If not, see <http://www.gnu.org/licenses/>.

from os import path
import numpy as np

def load_raw(name):
    """ Load a GRay2 raw file """
    ext = path.splitext(name)[1][1:]
    if ext != "raw":
        raise NameError("Fail to load file \"{}\", "
                        "which is in an unsupported format".format(name))

    with open(name, "rb") as f:
        print("Loading GRay2 raw file \"{}\"... ".format(name), end="")

        d = np.fromfile(f, dtype=np.uint64, count=4)
        t = np.double if d[0] == 8 else np.float
        n = d[1]
        w = d[2]
        h = d[3]

        states = np.fromfile(f, dtype=t, count=n*w*h).reshape((h,w,n))
        diagno = np.fromfile(f, dtype=t, count=  w*h).reshape((h,w  ))

        print("DONE")

        return states, diagno
