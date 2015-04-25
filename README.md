`GRay` is a massive parallel ordinary differential equation
integrator.  It employs the "stream processing paradigm" and runs on
nVidia's GPUs.  It is designed to efficiently integrate millions of
photons in curved spacetime according to Einstein's general theory of
relativity.


Compile the Code
----------------

To get started, simply type `make` in the `gray/` directory.  If the
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

The `Makefile` will then compile GRay and place the executable in
bin/.  To run `GRay` so it creates snapshot every 100 GM/c^3, type

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

Note that the `dt` parameter must be negative because in ray tracing
we integrate the rays backward.  The snapshot files "00.raw",
"01.raw", "02.raw", ..., named by the `snapshot` parameter contains
the full state dump of the run.  The final output "out.raw" is
controlled by `src/Kerr/output.cc`, and is currently used to output a
ray tracing image.


Code Structure
--------------

GRay is implemented in CUDA C/C++.  The following chart illustrates
the structure and flow of GRay:

    main() in "main.cc"
    |
    +-initialize GLUT and create window if OpenGL is enabled
    |
    +-parse arguments
    |
    +-Data::Data() in "data.cc", allocate memory on host and device
    |
    +-Data::init() in "init.cc", includes "*/init.h", initialize data
    |
    +-vis() in "vis.cc" setup visualization if OpenGL is enabled
    | |
    | +-mkshaders() in "shaders.cc"
    | |
    | +-mktexture() in "texture.cc"
    | |
    | +-setup OpenGL
    | |
    | +-regctrl() in "ctrl.cc"
    | |
    | +-register display() and reshape(); display() needs getctrl() in
    |   "ctrl.cc"
    |
    +-solve() in "solve.cc"
      |
      +-dump() in "io.cc" output initial conditions if IO is enabled
      |
      +-mainloop in "solve.cu" by calling back idle() if OpenGL is---<---<---<-+
        enabled or by calling evolve() directly in a while loop;               |
        the idle() function is a smart wrapper that adjust dt_ump              |
        |                                                                      |
        +-evolve() in "evolve.cu"                                              ^
        | |                                                                    |
        | +-register cleanup() in "setup.cc"                                   ^
        | | |                                                                  |
        | | +-free counter and destroy timers                                  ^
        | |                                                                    |
        | +-allocate counter and create CUDA timers                            ^
        | |                                                                    |
        | +-driver() global function in "driver.cu"                            ^
        |   |                                                                  |
        |   +-substep while loop--<---<---<---<---<---<---<---<---<---<---<-+  ^
        |     |                                                             |  |
        |     +-scheme() device function in "scheme/*.cu"                   ^  ^
        |       |                                                           |  |
        |       +-getdt() device function in "*/getdt.cu"                   ^  ^
        |       |                                                           |  |
        |       +-rhs() device function in "*/rhs.cu"->--->--->--->--->--->-+  ^
        |                                                                      |
        +-dump() in "io.cc", see above                                         ^
        |                                                                      |
        +-call the control call-backs; request graphics update and loop back->-+
