GRay is a massive parallel ordinary differential equation integrator.
It employs the *stream processing paradigm* and runs on nVidia's
Graphics Processing Units (GPUs).  It is designed to efficiently
integrate millions of photons in curved spacetime according to
Einstein's general theory of relativity.


Compile the Code
----------------

To get started, simply type `make` in the gray/ directory.  If the
nVidia CUDA compiler `nvcc` is in your path, you will see the
following instructions:

    gray$ make
    The follow problems are available:

      1. Kerr
      ...

    Use `make <prob> [DEBUG=1] [DETAILS=1] [DOUBLE/SINGLE=1] [GL=0]` and
    `bin/GRay-<prob>` to compile and to run GRay.  The option DEBUG=1
    turns on debugging messages, DETAILS=1 prints ptxas information,
    ...

To solve the geodesic equations in the Kerr metric with interactive
mode disabled and in single-precision, simply type

    gray$ mk Kerr GL=0 SINGLE=1
    Compiling Kerr... DONE.  Use `bin/GRay-Kerr` to run GRay.

The Makefile will then compile GRay and place the executable in bin/.
To run GRay so it creates snapshot every 100 GM/c^3, type

    gray$ bin/GRay-Kerr snapshot=%02d.raw dt=-100
    GRay: a massive parallel ODE integrator written in CUDA C/C++
    Press 'Ctrl+C' to quit
    Set parameter "snapshot=%02d.raw"
    Set parameter "dt=-100"
    1 GPU is found --- running on GPU 0
    "GeForce GT 650M" with 1023.69MiB global and 48KiB shared memory
    t = -100.00; 64 ms/1048576 steps ~ 10.49 Gflops (100.00%), 3.96 GB/s
    t = -200.00; 64 ms/1048576 steps ~ 10.50 Gflops (100.00%), 3.96 GB/s
    t = -300.00; 79 ms/1310720 steps ~ 10.58 Gflops (100.00%), 3.99 GB/s
    ...

Note that the dt parameter must be negative because in ray tracing we
integrate the rays backward.  The snapshot files "00.raw", "01.raw",
"02.raw", ..., named by the snapshot parameter contains the full state
dump of the run.  The final output "out.raw" is controlled by
src/Kerr/output.cc, and is currently used to output a ray tracing
image.


Code Structure
--------------

GRay is implemented in CUDA C/C++.  The following chart illustrates
the structure and flow of GRay:

    main() in "main.cc"
    |
    +--Construct para using Para::Para() in "para.cc", which calls
    |  Para::define() in "*/config.cc" to set the default parameters
    |
    +--Parse command line arguments and uses Para::config() to pass
    |  the results to para
    |
    +--Construct data using Data::Data() in "data.cc", which allocates
    |  memory on both host and device
    |
    +--Call Data::init(), which is declared in "core.cu" and includes
    |  "*/ic.h" to initialize data
    |
    +--Dump initial condition using Data::snapshot() in "io.cc" if
    |  format is provided
    |
    +--Main loop: call Data::solve() in "solve.cc" in a while loop<---<---<-.
    |  |                                                                    ^
    |  +--Compute sub step size if GL is enabled                            |
    |  |                                                                    |
    |  +--Call Data::evolve() in "core.cu"                                  ^
    |  |                                                                    |
    |  +--Print benchmark result                                            |
    |  |                                                                    ^
    |  +--Create snapshot using Data::snapshot() in "io.cc" if format       |
    |  |  is provided                                                       |
    |  |                                                                    ^
    |  +-->--->--->--->--->--->--->--->--->--->--->--->--->--->--->--->--->-`
    |
    +--Create final output by calling Data::output() in "io.cc", which
       uses a different Data::output() in "*/io.cc"
