/*
 * Copyright (C) 2021 Gabriele Bozzola
 *
 * This file is part of GRay2.
 *
 * GRay2 is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GRay2 is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
 * License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GRay2.  If not, see <https://www.gnu.org/licenses/>.
 */

/** \file
 ** Interpolate fisheye coordinates.
 **
 ** To study black holes in dynamical spacetimes, one needs enough resolution
 ** near the horizons.  This is typically reported as a fraction of the mass of
 ** the black hole, for example a value of M/80 is the resolution that roughly
 ** covers the radius of a non-spinning black hole in the puncture gauge with 80
 ** points.  It is impossible to cover the entire numerical grid with such
 ** resolution, and possible solutions are mesh-refinement of fisheye
 ** coordinates.  This latter solution consists in deforming the coordinate
 ** system in such a way that the points are more concentrated near the horizons
 ** and less outside.  This can be achieved sampling the relevant functions in a
 ** non uniform way.  The fisheye transformation that we use here is:
 ** \f[
 **  x = \sinh \left[ {\left( A \xi + B \right)}^{n} \right]\,,
 ** \f]
 ** with \f$x\f$ physical coordinates and \f$\xi\f$ logically-Cartesian
 ** coordinates.  The parameter \f$n\f$ determines how much concentrated are
 ** points near the center, and the other two parameters are fixed by fixing the
 ** extent that \f$x\f$ has to cover.
 **
 ** When fisheye coordinates are used, it is critical to correct how OpenCL
 ** performs the multilinear interpolation to account for the fact that points
 ** are distributed unevenly.  This module provides the infrastructure to do
 ** that.  Additionally, it also handles time interpolation (with a linear
 ** transformation).
 **
 **/

/* In this module we call xyz the physical coordinates and uvw the
 * unnormalized, OpenCL ones.  The values are always stored in the .s123 slots
 * of the 4-vectors.  The slot .s0 is not used. */

inline int4 address_mode(int4 x, int4 size){
  /* We implement CLK_ADDRESS_CLAMP_TO_EDGE */

  /* We need this function because we need to convert from physical variables to
   * unnormalized ones. */
  return clamp(x, (int4){0,0,0,0}, size - 1);
}

inline real4 address_mode_real(real4 x, real4 size){
  /* We implement CLK_ADDRESS_CLAMP_TO_EDGE */

  /* We need this function because we need to convert from physical variables to
   * unnormalized ones. */
  return clamp(x, (real4){K(0.0), K(0.0), K(0.0), K(0.0)}, size - 1);
}

int4 xyz_to_uvw(real4 xyz, real8 bounding_box, int4 num_points){
  /* Returns the unnormalized point uvw corresponding to xyz (rounding down) */

  /* FISHEYE IS HERE */
  /* In this function, we compute the coordinate transformation from physical
   * coordinates to unnormalized ones.  This does not take into account the
   * fact that coordinates are unevenly spaced, and it simply consists in the
   * application of the fisheye transformation. */

  /* Xmin = {0, xmin, ymin, zmin} */
  /* Xmax = {0, xmax, ymax, zmax} */
  real4 Xmin = {K(0.0), bounding_box.s1, bounding_box.s2, bounding_box.s3};
  real4 Xmax = {K(0.0), bounding_box.s5, bounding_box.s6, bounding_box.s7};

  /* num_points_real is defined like this because there is no easy way to cast a
   * vector to a real4, so we instead define a new variables where we cast the
   * individual variables. */
  real4 num_points_real = {num_points.s0, num_points.s1, num_points.s2,
                           num_points.s3};

  /* The fisheye transformation is hard-coded here */
  /* We work with n = 3 (n is the exponent in the sinh) */
  real4 B = cbrt(asinh(Xmin));
  real4 A = (cbrt(asinh(Xmax)) - B)/(num_points_real - 1);

  real4 xi = floor((cbrt(asinh(xyz)) - B)/A);

  return address_mode(convert_int4(xi), num_points);
}

real4 uvw_to_xyz(int4 uvw, real8 bounding_box, real4 num_points_real){

  /* Returns the physical point xyz corresponding to unnormalized uvw */

  /* FISHEYE IS HERE */

  /* Xmin = {0, xmin, ymin, zmin} */
  /* Xmax = {0, xmax, ymax, zmax} */
  real4 Xmin = {K(0.0), bounding_box.s1, bounding_box.s2, bounding_box.s3};
  real4 Xmax = {K(0.0), bounding_box.s5, bounding_box.s6, bounding_box.s7};

  /* uvw_real is defined like this because there is no easy way to cast a
   * vector to a real4, so we instead define a new variables where we cast the
   * individual variables. */
  real4 uvw_real = {uvw.s0, uvw.s1, uvw.s2, uvw.s3};

  /* Hardcode the coordinate transformation with n = 3 */
  real4 B = cbrt(asinh(Xmin));
  real4 A = (cbrt(asinh(Xmax)) - B)/(num_points_real - 1);

  real4 fact = (A * uvw_real + B);
  return sinh(fact * fact * fact);
}

real4 find_correct_uvw(real4 xyz,
                       real8 bounding_box,
                       int4 num_points){

  /* Return the OpenCL unnormalized coordinates uvw that, if plugged in the
   * multilinear interpolation routines, would return the correct interpolated
   * value for the physical coordinate xyz. */

  /* To do this, we first need to find the unnormalized coordinates that bound
   * the given physical points xyz.  We call these uvw_ijk and uvw_ijkp1.  The
   * "p1" means "plus_one" as we know that uvw_ijk will be the lower edge.
   * Then, we compute the corresponding physical coordinates and we perform
   * linear interpolation between the two. */

  /* num_points_real is defined like this because there is no easy way to cast a
   * vector to a real4, so we instead define a new variables where we cast the
   * individual variables. */
  real4 num_points_real = {num_points.s0, num_points.s1, num_points.s2,
                           num_points.s3};

  int4 uvw_ijk   = xyz_to_uvw(xyz, bounding_box, num_points);
  int4 uvw_ijkp1 = uvw_ijk + 1;

  real4 xyz_ijk   = uvw_to_xyz(uvw_ijk,   bounding_box, num_points_real);
  real4 xyz_ijkp1 = uvw_to_xyz(uvw_ijkp1, bounding_box, num_points_real);

  /* uvw_ijk_real is defined like this because there is no easy way to cast a
   * vector to a real4, so we instead define a new variables where we cast the
   * individual variables. */
  real4 uvw_ijk_real = {uvw_ijk.s0, uvw_ijk.s1, uvw_ijk.s2, uvw_ijk.s3};

  /* Linear interpolation of coordinates*/
  real4 uvw_interp = uvw_ijk_real + (xyz - xyz_ijk)/(xyz_ijkp1 - xyz_ijk);

  /* We clamp to edge, to make sure we are not producing values that are outside
   * the range of definition of the data */
  uvw_interp = address_mode_real(uvw_interp, num_points_real);

  /* Finally, we have to offset by 0.5.  This 0.5 is very important because OpenCL
   * uses a pixel offset of 0.5 */
  return uvw_interp + (real4){K(0.5), K(0.5), K(0.5), K(0.5)};
}

real space_interpolate(real4 xyz,
                       real8 bounding_box,
                       int4 num_points,
                       __read_only image3d_t var){

  /* Return var evaluated on xyz */
  sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE \
                      | CLK_FILTER_LINEAR;

  real4 coords = find_correct_uvw(xyz, bounding_box, num_points);
  /* In read_imagef, coords have to be in the slots 0, 1, and 2. The slot 3 is
   * ignored */
  coords.s012 = coords.s123;

  return read_imagef(var, sampler, coords).x;
}

real time_interpolate(real t, real t1, real t2, real y1, real y2){
  /* Perform linear time interpolation between t1 and t2 with values y1 and
   * y2. */

  /* y(t) = y_1 + (t - t_1) / (t_2 - t_1) * (y_2 - y_1) */

  return y1 + (t - t1) / (t2 - t1) * (y2 - y1);
}

real interpolate(real4 q,
                 real8 bounding_box,
                 int4 num_points,
                 __read_only image3d_t var_t1,
                 __read_only image3d_t var_t2){

  /* Return var interpolated on q.  If the bounding box is defined on a single
   * time level, then use only var_t1, otherwise perform linear interpolation
   * in time between var_t1 and var_t2. */

  real t1 = bounding_box.s0;
  real t2 = bounding_box.s4;

  if (t1 == t2)
    return space_interpolate(q, bounding_box, num_points, var_t1);

  real y1 = space_interpolate(q, bounding_box, num_points, var_t1);
  real y2 = space_interpolate(q, bounding_box, num_points, var_t2);

  return time_interpolate(q.s0, t1, t2, y1, y2);
}
