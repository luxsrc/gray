- The nVidia CUDA compiler currently fails to compile GRay in
  double-precision mode.  Find a work-around.

- Resample HARM's GRMHD data to a nicer grid so that the indices can
  be easily computed.

- Use texture memory and hardware interpolation to access HARM's GRMHD
  data.

- Automatically split the image if the number of rays/pixels is too
  large (e.g., 8192 x 8192).
