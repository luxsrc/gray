# Copyright (C) 2012 Chi-kwan Chan
# Copyright (C) 2012 Steward Observatory
#
# This file is part of geode.
#
# Geode is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Geode is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with geode.  If not, see <http://www.gnu.org/licenses/>.

import numpy             as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

def load(i):
    import struct
    f  = open(str(i).zfill(4)+'.raw', 'rb')
    t, = struct.unpack('d', f.read(8))
    m, = struct.unpack('i', f.read(4))
    n, = struct.unpack('i', f.read(4))
    d  = np.fromfile(f, np.dtype('f4'), m * n).reshape((n, m))

    if m == 6:
        print t
        return {'t':t, 'x':d[:,0], 'y':d[:,1], 'z':d[:,2]}
    else:
        t     = d[:,0]
        r     = d[:,1]
        theta = d[:,2]
        phi   = d[:,3]

        r_cyl = r     * np.sin(theta)
        x     = r_cyl * np.cos(phi  )
        y     = r_cyl * np.sin(phi  )
        z     = r     * np.cos(theta)

        print min(t), max(t)
        return {'t':t, 'x':x, 'y':y, 'z':z}

n = 16
m = 16

x = np.zeros((n, m))
y = np.zeros((n, m))
z = np.zeros((n, m))
for i in range(m):
    l = load(i)
    x[:,i] = l['x'][0:n]
    y[:,i] = l['y'][0:n]
    z[:,i] = l['z'][0:n]

ax = plt.figure().gca(projection='3d')

for i in range(n):
    plt.plot(x[i,:], y[i,:], z[i,:])

ax.set_xlim(-10,10)
ax.set_ylim(-10,10)
ax.set_zlim(-10,10)

plt.show()
