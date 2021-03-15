#!/usr/bin/env python3

# Copyright (C) 2021 Pierre Christian, Gabriele Bozzola
# Copyright (C) 2020 Pierre Christian
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

############## How to use ##################################
"""
Step 1) Check that Pierre wrote down metric/Christoffel symbols correctly

Step 2) Compute Christoffel symbol Gamma^i_kl using Christoffel(Param,Coord,i,k,l). Param=parameters of the metric; for the KS metric, Param=[Mass,spin]. Coord=coordinates; for the KS metric, Coord=[time,x,y,z]. i, k, and l are spacetime indices; they range from 0-3 (Yes, I use i for something that ranges from 0-3)

Other functions:
-) metric(Param,Coord,i,j,updown) gives the metric. The switch updown is either "up" for contravariant metric or "down" for covariant metric. Again, despite how they appear, i and j are spacetime indices (ranges from 0-3).

-) dm(Param,Coord,i,j,wrt,updown) gives the metric derivatives. The variable wrt specifies that the derivative is taken "with respect to" which coordinate. For example, dm(Param,Coord,i,j,0,updown) is the metric derivative w.r.t. time.

"""

############## import all sorts of things, most of them not used ##################################

import numpy as np
from functools import lru_cache

############## Dual number stuff ###########################


class dual:
    def __init__(self, first, second):
        self.f = first
        self.s = second

    @lru_cache(1024)
    def __mul__(self, other):
        if isinstance(other, dual):
            return dual(self.f * other.f, self.s * other.f + self.f * other.s)
        else:
            return dual(self.f * other, self.s * other)

    @lru_cache(1024)
    def __rmul__(self, other):
        if isinstance(other, dual):
            return dual(self.f * other.f, self.s * other.f + self.f * other.s)
        else:
            return dual(self.f * other, self.s * other)

    @lru_cache(1024)
    def __add__(self, other):
        if isinstance(other, dual):
            return dual(self.f + other.f, self.s + other.s)
        else:
            return dual(self.f + other, self.s)

    @lru_cache(1024)
    def __radd__(self, other):
        if isinstance(other, dual):
            return dual(self.f + other.f, self.s + other.s)
        else:
            return dual(self.f + other, self.s)

    @lru_cache(1024)
    def __sub__(self, other):
        if isinstance(other, dual):
            return dual(self.f - other.f, self.s - other.s)
        else:
            return dual(self.f - other, self.s)

    @lru_cache(1024)
    def __rsub__(self, other):
        return dual(other, 0) - self

    @lru_cache(1024)
    def __truediv__(self, other):
        """ when the first component of the divisor is not 0 """
        if isinstance(other, dual):
            return dual(
                self.f / other.f,
                (self.s * other.f - self.f * other.s) / (other.f ** 2.0),
            )
        else:
            return dual(self.f / other, self.s / other)

    @lru_cache(1024)
    def __rtruediv__(self, other):
        return dual(other, 0).__truediv__(self)

    @lru_cache(1024)
    def __neg__(self):
        return dual(-self.f, -self.s)

    @lru_cache(1024)
    def __pow__(self, power):
        return dual(self.f ** power, self.s * power * self.f ** (power - 1))

@lru_cache(1024)
def dif(func, x):
    funcdual = func(dual(x, 1.0))

    if isinstance(funcdual, dual):
        return func(dual(x, 1.0)).s

    # This is for when the function is a constant, e.g. gtt:=0
    return 0


################### Metric #####################################

def metric(Param, Coord, i, j, up=False,
           Kerr_KerrSchild__t=0, Kerr_KerrSchild__x=0,
           Kerr_KerrSchild__y=0, Kerr_KerrSchild__z=0):

    m, a, boostv = Param
    t, x, y, z = Coord

    gamma = 1 / (1 - boostv**2)**0.5

    t0 = gamma * ((t - Kerr_KerrSchild__t) - boostv * (z - Kerr_KerrSchild__z))
    z0 = gamma * ((z - Kerr_KerrSchild__z) - boostv * (t - Kerr_KerrSchild__t))
    x0 = x - Kerr_KerrSchild__x
    y0 = y - Kerr_KerrSchild__y

    rho02 = x0**2 + y0**2 + z0**2

    r02 = 0.5 * (rho02 - a**2) + (0.25 * (rho02 - a**2)**2 + a**2 * z0**2)**0.5
    r0 = r02**0.5
    costheta0 = z0 / r0

    hh = m * r0 / (r0**2 + a**2 * costheta0**2)

    lt0 = 1
    lx0 = (r0 * x0 + a * y0) / (r0**2 + a**2)
    ly0 = (r0 * y0 - a * x0) / (r0**2 + a**2)
    lz0 = z0 / r0

    lt = gamma * (lt0 - boostv * lz0)
    lz = gamma * (lz0 - boostv * lt0)
    lx = lx0
    ly = ly0

    gdtt = - 1 + 2 * hh * lt * lt
    gdtx = 2 * hh * lt * lx
    gdty = 2 * hh * lt * ly
    gdtz = 2 * hh * lt * lz
    gdxx = 1 + 2 * hh * lx * lx
    gdyy = 1 + 2 * hh * ly * ly
    gdzz = 1 + 2 * hh * lz * lz
    gdxy = 2 * hh * lx * ly
    gdyz = 2 * hh * ly * lz
    gdzx = 2 * hh * lz * lx

    gutt = - 1 - 2 * hh * lt * lt
    gutx = 2 * hh * lt * lx
    guty = 2 * hh * lt * ly
    gutz = 2 * hh * lt * lz
    guxx = 1 - 2 * hh * lx * lx
    guyy = 1 - 2 * hh * ly * ly
    guzz = 1 - 2 * hh * lz * lz
    guxy = - 2 * hh * lx * ly
    guyz = - 2 * hh * ly * lz
    guzx = - 2 * hh * lz * lx

    g_down = [[gdtt, gdtx, gdty, gdtz],
              [gdtx, gdxx, gdxy, gdzx],
              [gdty, gdxy, gdyy, gdyz],
              [gdtz, gdzx, gdyz, gdzz]]

    g_up = [[gutt, gutx, guty, gutz],
            [gutx, guxx, guxy, guzx],
            [guty, guxy, guyy, guyz],
            [gutz, guzx, guyz, guzz]]

    if up:
        return g_up[i][j]
    return g_down[i][j]


##################### Metric derivatives #############################

@lru_cache(1024)
def dm(Param, Coord, i, j, wrt, up=False):
    """ This computes metric derivatives. wrt = 0,1,2,3 is derivative "with respect to" which coordinate; i,j are spacetime indices. (Yes, I use i and j for something that range from 0-3) """
    point_d = Coord[wrt]

    point_0 = dual(Coord[0], 0)
    point_1 = dual(Coord[1], 0)
    point_2 = dual(Coord[2], 0)
    point_3 = dual(Coord[3], 0)

    if wrt == 0:
        return dif(
            lambda p: metric(
                Param, (p, point_1, point_2, point_3), i, j, up
            ),
            point_d,
        )
    elif wrt == 1:
        return dif(
            lambda p: metric(
                Param, (point_0, p, point_2, point_3), i, j, up
            ),
            point_d,
        )
    elif wrt == 2:
        return dif(
            lambda p: metric(
                Param, (point_0, point_1, p, point_3), i, j, up
            ),
            point_d,
        )
    elif wrt == 3:
        return dif(
            lambda p: metric(
                Param, (point_0, point_1, point_2, p), i, j, up
            ),
            point_d,
        )


##################### Christoffel Symbols #############################

@lru_cache(1024)
def Chris_anc_A(Param, Coord, i, m, k, l):
    return (
        metric(Param, Coord, i, m, up=True)
        * dm(Param, Coord, m, k, l)
    )

@lru_cache(1024)
def Chris_anc_B(Param, Coord, i, m, k, l):
    return (
        metric(Param, Coord, i, m, up=True)
        * dm(Param, Coord, m, l, k)
    )

@lru_cache(1024)
def Chris_anc_C(Param, Coord, i, m, k, l):
    return (
        metric(Param, Coord, i, m, up=True)
        * dm(Param, Coord, k, l, m)
    )

def Christoffel(Param, Coord, i, k, l):
    """ Gamma^i_kl """
    Term1 = (
        Chris_anc_A(Param, Coord, i, 0, k, l)
        + Chris_anc_A(Param, Coord, i, 1, k, l)
        + Chris_anc_A(Param, Coord, i, 2, k, l)
        + Chris_anc_A(Param, Coord, i, 3, k, l)
    )
    Term2 = (
        Chris_anc_B(Param, Coord, i, 0, k, l)
        + Chris_anc_B(Param, Coord, i, 1, k, l)
        + Chris_anc_B(Param, Coord, i, 2, k, l)
        + Chris_anc_B(Param, Coord, i, 3, k, l)
    )
    Term3 = (
        Chris_anc_C(Param, Coord, i, 0, k, l)
        + Chris_anc_C(Param, Coord, i, 1, k, l)
        + Chris_anc_C(Param, Coord, i, 2, k, l)
        + Chris_anc_C(Param, Coord, i, 3, k, l)
    )

    return 0.5 * (Term1 + Term2 - Term3)
