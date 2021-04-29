# GRay2 Documentation {#mainpage}

GRay2 is a hardware-accelerated geodesic integrator for performing
general relativistic ray tracing for accreting black holes.
It is based on the [lux framework](https://luxsrc.org) and runs on a
wide range of modern hardware/accelerators such as GPUs and Intel&reg;
Xeon Phi.

## For users

For people we are interested in using GRay2 as-is, please download a
tarball from GitHub's release page:

	https://github.com/luxsrc/gray/releases

Assuming `lux` is installed (see https://github.com/luxsrc/lux), users
can simply run

	make

to build GRay2 as a `lux` job and run GRay2 by `lux gray`.

## For developers

For people we are interested in contributing to GRay2, please fork
GRay2's git repository on GitHub:

	https://github.com/luxsrc/gray

work on your fork, and then create pull request to merge your changes
back to the main repository.

GRay2 is flexible and easily extendable.
To turn hard-wired constants into run-time options, follow the
instructions in \ref newopts "this page".
To add new computation kernels to GRay2, see \ref newkern "this page".
We also keep track of a list of TODOs found in the code \ref todo
"here".

## Structure of HDF5 files

GRay2 can read spacetime and fluid data from HDF5 files. These files must
be structured in a specific way:
* At the top level, they must contain a group called "grid". This group has to
  contain three datasets named "x", "y", "z", which contains the coordinates
  along the three directions.
* Always at the top level, all the groups that are not named "grid" will be
  considered time levels. The names of such groups has to be their time. For
  example, possible groups names would be "1.0", "1.1", "1.2", ... The group
  "1.0" contains variables at that time. The groups have to be in alphanumerical
  order.
* In each group, the following datasets have to be defined. Gamma_ttt,
  Gamma_ttx, Gamma_tty, Gamma_ttz, Gamma_txx, Gamma_txy, Gamma_txz, Gamma_tyy,
  Gamma_tyz, Gamma_tzz, Gamma_xtt, Gamma_xtx, Gamma_xty, Gamma_xtz, Gamma_xxx,
  Gamma_xxy, Gamma_xxz, Gamma_xyy, Gamma_xyz, Gamma_xzz, Gamma_ytt, Gamma_ytx,
  Gamma_yty, Gamma_ytz, Gamma_yxx, Gamma_yxy, Gamma_yxz, Gamma_yyy, Gamma_yyz,
  Gamma_yzz, Gamma_ztt, Gamma_ztx, Gamma_zty, Gamma_ztz, Gamma_zxx, Gamma_zxy,
  Gamma_zxz, Gamma_zyy, Gamma_zyz, Gamma_zzz, g_tt, g_tx, g_ty, g_tz, g_xx,
  g_xy, g_xz, g_yy, g_yz, g_zz,
* All the variables must have the same precision (e.g., single or double).
* Each group has to be tagged with attributes with names ah_N, where N is the
  number of horizon (starting from 1). These attributes have to be arrays with
  5 elements: the x,y,z location of the Nth horizon, and its minimum and
  maximum radii.
