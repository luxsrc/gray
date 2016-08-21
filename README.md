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
To add new computation kernels to GRya2, see \ref newkern "this page".
We also keep track of a list of TODOs found in the code \ref todo
"here".
