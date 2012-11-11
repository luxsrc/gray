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
    n, = struct.unpack('i', f.read(4))
    d  = np.fromfile(f, np.dtype('f4'), 6 * n).reshape((n, 6))
    print t
    return {'t':t, 'x':d[:,0], 'y':d[:,1], 'z':d[:,2]}

n = 16
m = 100

x = np.zeros((n, m))
y = np.zeros((n, m))
z = np.zeros((n, m))
for i in range(m):
    l = load(i)
    x[:,i] = l['x'][0:n]
    y[:,i] = l['y'][0:n]
    z[:,i] = l['z'][0:n]

plt.figure().gca(projection='3d')
for i in range(n):
    plt.plot(x[i,:], y[i,:], z[i,:])
plt.show()
