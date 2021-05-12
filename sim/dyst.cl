/* Automatically generated, do not edit */

#define compute_eps_hor(index)                                          \
  if (ah_valid_##index[snap_number] > 0){                               \
                                                                        \
    /* Perform time interpolation for radius and centroid */            \
                                                                        \
    /* Note that we don't allow runs where t_final is outside the range of \
     * snapshots, so we always have a "+1" snapshot. When we are working with \
     * only one snapshot, we copied over the horizons to have two, so that \
     * we can still run through these routines. */                      \
                                                                        \
    /* We do not want to include the time in the distance, so we set    \
     * it to be the same as out point. */                               \
    centroid.s0 = q.s0;                                                 \
                                                                        \
    /* We need to do time interpolation only if the times are different \
     * Otherwise, we can just read the value */                         \
                                                                        \
    if (ah_centr_##index[snap_number].s0                                \
        != ah_centr_##index[snap_number + 1].s0) {                      \
                                                                        \
    centroid.s1 = time_interpolate(                                     \
      q.s0,                                                             \
      ah_centr_##index[snap_number].s0,       /* t1 */                  \
      ah_centr_##index[snap_number + 1].s0,   /* t2 */                  \
      ah_centr_##index[snap_number].s1,       /* y1 */                  \
      ah_centr_##index[snap_number + 1].s1);  /* y2 */                  \
                                                                        \
    centroid.s2 = time_interpolate(                                     \
      q.s0,                                                             \
      ah_centr_##index[snap_number].s0,       /* t1 */                  \
      ah_centr_##index[snap_number + 1].s0,   /* t2 */                  \
      ah_centr_##index[snap_number].s2,       /* y1 */                  \
      ah_centr_##index[snap_number + 1].s2);  /* y2 */                  \
                                                                        \
    centroid.s3 = time_interpolate(                                     \
      q.s0,                                                             \
      ah_centr_##index[snap_number].s0,       /* t1 */                  \
      ah_centr_##index[snap_number + 1].s0,   /* t2 */                  \
      ah_centr_##index[snap_number].s3,       /* y1 */                  \
      ah_centr_##index[snap_number + 1].s3);  /* y2 */                  \
                                                                        \
    radius_max = time_interpolate(                                      \
      q.s0,                                                             \
      ah_centr_##index[snap_number].s0,       /* t1 */                  \
      ah_centr_##index[snap_number + 1].s0,   /* t2 */                  \
      ah_max_r_##index[snap_number],          /* y1 */                  \
      ah_max_r_##index[snap_number + 1]);     /* y2 */                  \
    }else{                                                              \
      centroid.s1 = ah_centr_##index[snap_number].s1;                   \
      centroid.s2 = ah_centr_##index[snap_number].s2;                   \
      centroid.s3 = ah_centr_##index[snap_number].s3;                   \
      radius_max = ah_max_r_##index[snap_number];                       \
    }                                                                   \
                                                                        \
    eps_tmp = sqrt(distance(q, centroid)) - radius_max;                 \
    if (eps_tmp < eps) eps = eps_tmp;                                   \
}

struct gr {
    real4 q;
    real4 u;
};

inline real GRAY_SQUARE (real x) { return x*x; };
inline real GRAY_CUBE (real x) { return x*x*x; };
inline real GRAY_FOUR (real x) { return x*x*x*x; };
inline real GRAY_SQRT (real x) { return sqrt(x); };
inline real GRAY_SQRT_CUBE (real x) { return sqrt(x*x*x); };

inline int is_q_inside_boundary (real4 q, real8 bounding_box) {
  /* If all the inequalities are satisfied, then inside_boundary is 1, if one
   * or more are not, then it will be 0. */
  return (q.s1 < bounding_box.s5) * (q.s1 > bounding_box.s1) \
    * (q.s2 < bounding_box.s6) * (q.s2 > bounding_box.s2) \
    * (q.s3 < bounding_box.s7) * (q.s3 > bounding_box.s3);
};

real16 matrix_product(real16 a, real16 b){

  real4 a_row0 = a.s0123;
  real4 a_row1 = a.s4567;
  real4 a_row2 = a.s89ab;
  real4 a_row3 = a.scdef;
  real4 b_col0 = b.s048c;
  real4 b_col1 = b.s159d;
  real4 b_col2 = b.s26ae;
  real4 b_col3 = b.s37bf;

  return (real16){dot(a_row0, b_col0), dot(a_row0, b_col1),
  dot(a_row0, b_col2), dot(a_row0, b_col3),
  dot(a_row1, b_col0), dot(a_row1, b_col1),
  dot(a_row1, b_col2), dot(a_row1, b_col3),
  dot(a_row2, b_col0), dot(a_row2, b_col1),
  dot(a_row2, b_col2), dot(a_row2, b_col3),
  dot(a_row3, b_col0), dot(a_row3, b_col1),
  dot(a_row3, b_col2), dot(a_row3, b_col3)};
};

real4 matrix_vector_product(real16 a, real4 b){

  return (real4){dot(a.s0123, b),
  dot(a.s4567, b),
  dot(a.s89ab, b),
  dot(a.scdef, b)};
};

real
getrr(real4 q)
{
  /* We use the flat Cartesian metric for this. This is used
   * only to check if we are far away from the center of the
   * domain. */
  return q.s1 * q.s1 + q.s2 * q.s2 + q.s3 * q.s3;
}

real
geteps(real4 q, const whole snap_number,
       HORIZON_PROTOTYPE_ARGS)
{

  /* We loop over the horizons, look at the ones that are valid,
   * and take the Euclidean distance from a sphere defined by
   * the minimum radius. Then, we take the minimum across all
   * the horizons. If there are no horizons, then we set eps
   * to a very large value. */

  real eps = K(1e4);
  real eps_tmp;

  real4 centroid;
  real radius_max;

  /* One per each horizon */
  compute_eps_hor(1);
  compute_eps_hor(2);
  compute_eps_hor(3);

  return eps;
}


void interp_metric(real4 q, real16 *g, SPACETIME_PROTOTYPE_ARGS)
{
  /* Performing the fisheye transformation is expensive. So, we are not going to
   * use the full interp.cl module, but part of it, and implement the
   * interpolation of all the Gammas at once. */

  /* We follow space_interpolate */

  /* Return var evaluated on xyz */
  sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE \
    | CLK_FILTER_LINEAR;

  real4 coords = find_correct_uvw(q, bounding_box, num_points);
  /* In read_imagef, coords have to be in the slots 0, 1, and 2. The slot 3 is
   * ignored */
  coords.s012 = coords.s123;

  real t1 = bounding_box.s0;
  real t2 = bounding_box.s4;
  real t  = q.s0;

  if (t1 == t2){
    (*g).s0 = interpolate(q, bounding_box, num_points, g_tt_t1, g_tt_t2);
    (*g).s1 = interpolate(q, bounding_box, num_points, g_tx_t1, g_tx_t2);
    (*g).s2 = interpolate(q, bounding_box, num_points, g_ty_t1, g_ty_t2);
    (*g).s3 = interpolate(q, bounding_box, num_points, g_tz_t1, g_tz_t2);
    (*g).s4 = (*g).s1;
    (*g).s5 = interpolate(q, bounding_box, num_points, g_xx_t1, g_xx_t2);
    (*g).s6 = interpolate(q, bounding_box, num_points, g_xy_t1, g_xy_t2);
    (*g).s7 = interpolate(q, bounding_box, num_points, g_xz_t1, g_xz_t2);
    (*g).s8 = (*g).s2;
    (*g).s9 = (*g).s6;
    (*g).sa = interpolate(q, bounding_box, num_points, g_yy_t1, g_yy_t2);
    (*g).sb = interpolate(q, bounding_box, num_points, g_yz_t1, g_yz_t2);
    (*g).sc = (*g).s3;
    (*g).sd = (*g).s7;
    (*g).se = (*g).sb;
    (*g).sf = interpolate(q, bounding_box, num_points, g_zz_t1, g_zz_t2);
  }else{
    (*g).s0 = time_interpolate(t, t1, t2,
                               read_imagef(g_tt_t1, sampler, coords).x,
                               read_imagef(g_tt_t2, sampler, coords).x);
    (*g).s1 = time_interpolate(t, t1, t2,
                               read_imagef(g_tx_t1, sampler, coords).x,
                               read_imagef(g_tx_t2, sampler, coords).x);
    (*g).s2 = time_interpolate(t, t1, t2,
                               read_imagef(g_ty_t1, sampler, coords).x,
                               read_imagef(g_ty_t2, sampler, coords).x);
    (*g).s3 = time_interpolate(t, t1, t2,
                               read_imagef(g_tz_t1, sampler, coords).x,
                               read_imagef(g_tz_t2, sampler, coords).x);
    (*g).s4 = (*g).s1;
    (*g).s5 = time_interpolate(t, t1, t2,
                               read_imagef(g_xx_t1, sampler, coords).x,
                               read_imagef(g_xx_t2, sampler, coords).x);
    (*g).s6 = time_interpolate(t, t1, t2,
                               read_imagef(g_xy_t1, sampler, coords).x,
                               read_imagef(g_xy_t2, sampler, coords).x);
    (*g).s7 = time_interpolate(t, t1, t2,
                               read_imagef(g_xz_t1, sampler, coords).x,
                               read_imagef(g_xz_t2, sampler, coords).x);
    (*g).s8 = (*g).s2;
    (*g).s9 = (*g).s6;
    (*g).sa = time_interpolate(t, t1, t2,
                               read_imagef(g_yy_t1, sampler, coords).x,
                               read_imagef(g_yy_t2, sampler, coords).x);
    (*g).sb = time_interpolate(t, t1, t2,
                               read_imagef(g_yz_t1, sampler, coords).x,
                               read_imagef(g_yz_t2, sampler, coords).x);
    (*g).sc = (*g).s3;
    (*g).sd = (*g).s7;
    (*g).se = (*g).sb;
    (*g).sf = time_interpolate(t, t1, t2,
                               read_imagef(g_zz_t1, sampler, coords).x,
                               read_imagef(g_zz_t2, sampler, coords).x);


  }
}


real4
down(real4 q, real4 u, SPACETIME_PROTOTYPE_ARGS)
{
  real16 g;

  int inside_boundary = is_q_inside_boundary(q, bounding_box);

  if (inside_boundary){
    interp_metric(q, &g, SPACETIME_ARGS);
  }else{
    real  rr = q.s1 * q.s1 + q.s2 * q.s2 + q.s3 * q.s3;
    real  r  = sqrt(rr);

    real  f  = K(2.0) * rr * r / (rr * rr);
    real  lx = q.s1 / r;
    real  ly = q.s2 / r;
    real  lz = q.s3 / r;

    real4 gt = {-1 + f   ,     f*   lx,     f*   ly,     f*   lz};
    real4 gx = {     f*lx, 1 + f*lx*lx,     f*lx*ly,     f*lx*lz};
    real4 gy = {     f*ly,     f*ly*lx, 1 + f*ly*ly,     f*ly*lz};
    real4 gz = {     f*lz,     f*lz*lx,     f*lz*ly, 1 + f*lz*lz};

    g.s0123 = gt;
    g.s4567 = gx;
    g.s89ab = gy;
    g.scdef = gz;
  }

  return matrix_vector_product(g, u);
}

real
getuu(struct gr s, SPACETIME_PROTOTYPE_ARGS)  /**< state of the ray */
{
  return dot(s.u, down(s.q, s.u, SPACETIME_ARGS));
}

struct gr
gr_icond(real r_obs, /**< Distance of the observer from the black hole */
         real i_obs, /**< Inclination angle of the observer in degrees */
         real j_obs, /**< Azimuthal   angle of the observer in degrees */
         real alpha, /**< One of the local Cartesian coordinates       */
         real beta,
         SPACETIME_PROTOTYPE_ARGS)  /**< The other  local Cartesian coordinate        */
{

  real  deg2rad = K(3.14159265358979323846264338327950288) / K(180.0);
  real  ci, si  = sincos(deg2rad * i_obs, &ci);
  real  cj, sj  = sincos(deg2rad * j_obs, &cj);

  real  R0 = r_obs * si - beta  * ci;
  real  z  = r_obs * ci + beta  * si;
  real  y  = R0    * sj + alpha * cj;
  real  x  = R0    * cj - alpha * sj;

  real4 q = (real4){K(0.0), x, y, z};
  real4 u = (real4){K(1.0), si * cj, si * sj, ci};

  real4 gt, gx, gy, gz;

  int inside_boundary = is_q_inside_boundary(q, bounding_box);

  if (inside_boundary){
    /* Here we are inside the boundary */

    real16 g;
    interp_metric(q, &g, SPACETIME_ARGS);
    gt = g.s0123;
    gx = g.s4567;
    gy = g.s89ab;
    gz = g.scdef;
  }else{
    /* We are outside the boundary, we use Kerr-Schild with spin = 0 */
    real  rr = q.s1 * q.s1 + q.s2 * q.s2 + q.s3 * q.s3;
    real  r  = sqrt(rr);

    real  f  = K(2.0) * rr * r / (rr * rr);
    real  lx = q.s1 / r;
    real  ly = q.s2 / r;
    real  lz = q.s3 / r;

    gt = (real4){-1 + f   ,     f*   lx,     f*   ly,     f*   lz};
    gx = (real4){     f*lx, 1 + f*lx*lx,     f*lx*ly,     f*lx*lz};
    gy = (real4){     f*ly,     f*ly*lx, 1 + f*ly*ly,     f*ly*lz};
    gz = (real4){     f*lz,     f*lz*lx,     f*lz*ly, 1 + f*lz*lz};
  }

	real  A  =  gt.s0;
	real  B  =  dot(gt.s123, u.s123) * K(2.0);
	real  C  = (dot(gx.s123, u.s123) * u.s1 +
	            dot(gy.s123, u.s123) * u.s2 +
	            dot(gz.s123, u.s123) * u.s3);

	u.s123 /= -(B + sqrt(B * B - K(4.0) * A * C)) / (K(2.0) * A);

  return (struct gr){q, u};
}

void interp_gammas(real4 q, real16 *GammaUPt, real16 *GammaUPx,
                   real16 *GammaUPy, real16 *GammaUPz,
                   SPACETIME_PROTOTYPE_ARGS)
{
  /* Performing the fisheye transformation is expensive. So, we are not going to
   * use the full interp.cl module, but part of it, and implement the
   * interpolation of all the Gammas at once. */

  /* We follow space_interpolate */

  /* Return var evaluated on q */
  sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE \
    | CLK_FILTER_LINEAR;

  real4 coords = find_correct_uvw(q, bounding_box, num_points);
  /* In read_imagef, coords have to be in the slots 0, 1, and 2. The slot 3 is
   * ignored */
  coords.s012 = coords.s123;

  real t1 = bounding_box.s0;
  real t2 = bounding_box.s4;
  real t  = q.s0;

  if (t1 == t2){
    (*GammaUPt).s0 = read_imagef(Gamma_ttt_t1, sampler, coords).x;
    (*GammaUPt).s1 = read_imagef(Gamma_ttx_t1, sampler, coords).x;
    (*GammaUPt).s2 = read_imagef(Gamma_tty_t1, sampler, coords).x;
    (*GammaUPt).s3 = read_imagef(Gamma_ttz_t1, sampler, coords).x;
    /* GammaUPt.s4 = GammaUPt.s1; */
    (*GammaUPt).s5 = read_imagef(Gamma_txx_t1, sampler, coords).x;
    (*GammaUPt).s6 = read_imagef(Gamma_txy_t1, sampler, coords).x;
    (*GammaUPt).s7 = read_imagef(Gamma_txz_t1, sampler, coords).x;
    /* GammaUPt.s8 = GammaUPt.s2; */
    /* GammaUPt.s9 = GammaUPt.s6; */
    (*GammaUPt).sa = read_imagef(Gamma_tyy_t1, sampler, coords).x;
    (*GammaUPt).sb = read_imagef(Gamma_tyz_t1, sampler, coords).x;
    /* GammaUPt.sc = GammaUPt.s3; */
    /* GammaUPt.sd = GammaUPt.s7; */
    /* GammaUPt.se = GammaUPt.sb; */
    (*GammaUPt).sf = read_imagef(Gamma_tzz_t1, sampler, coords).x;

    (*GammaUPt).s489 = (*GammaUPt).s126;
    (*GammaUPt).scde = (*GammaUPt).s37b;


    (*GammaUPx).s0 = read_imagef(Gamma_xtt_t1, sampler, coords).x;
    (*GammaUPx).s1 = read_imagef(Gamma_xtx_t1, sampler, coords).x;
    (*GammaUPx).s2 = read_imagef(Gamma_xty_t1, sampler, coords).x;
    (*GammaUPx).s3 = read_imagef(Gamma_xtz_t1, sampler, coords).x;
    /* GammaUPx.s4 = GammaUPx.s1; */
    (*GammaUPx).s5 = read_imagef(Gamma_xxx_t1, sampler, coords).x;
    (*GammaUPx).s6 = read_imagef(Gamma_xxy_t1, sampler, coords).x;
    (*GammaUPx).s7 = read_imagef(Gamma_xxz_t1, sampler, coords).x;
    /* GammaUPx.s8 = GammaUPx.s2; */
    /* GammaUPx.s9 = GammaUPx.s6; */
    (*GammaUPx).sa = read_imagef(Gamma_xyy_t1, sampler, coords).x;
    (*GammaUPx).sb = read_imagef(Gamma_xyz_t1, sampler, coords).x;
    /* GammaUPx.sc = GammaUPx.s3; */
    /* GammaUPx.sd = GammaUPx.s7; */
    /* GammaUPx.se = GammaUPx.sb; */
    (*GammaUPx).sf = read_imagef(Gamma_xzz_t1, sampler, coords).x;

    (*GammaUPx).s489 = (*GammaUPx).s126;
    (*GammaUPx).scde = (*GammaUPx).s37b;


    (*GammaUPy).s0 = read_imagef(Gamma_ytt_t1, sampler, coords).x;
    (*GammaUPy).s1 = read_imagef(Gamma_ytx_t1, sampler, coords).x;
    (*GammaUPy).s2 = read_imagef(Gamma_yty_t1, sampler, coords).x;
    (*GammaUPy).s3 = read_imagef(Gamma_ytz_t1, sampler, coords).x;
    /* GammaUPy.s4 = GammaUPy.s1; */
    (*GammaUPy).s5 = read_imagef(Gamma_yxx_t1, sampler, coords).x;
    (*GammaUPy).s6 = read_imagef(Gamma_yxy_t1, sampler, coords).x;
    (*GammaUPy).s7 = read_imagef(Gamma_yxz_t1, sampler, coords).x;
    /* GammaUPy.s8 = GammaUPy.s2; */
    /* GammaUPy.s9 = GammaUPy.s6; */
    (*GammaUPy).sa = read_imagef(Gamma_yyy_t1, sampler, coords).x;
    (*GammaUPy).sb = read_imagef(Gamma_yyz_t1, sampler, coords).x;
    /* GammaUPy.sc = GammaUPy.s3; */
    /* GammaUPy.sd = GammaUPy.s7; */
    /* GammaUPy.se = GammaUPy.sb; */
    (*GammaUPy).sf = read_imagef(Gamma_yzz_t1, sampler, coords).x;

    (*GammaUPy).s489 = (*GammaUPy).s126;
    (*GammaUPy).scde = (*GammaUPy).s37b;


    (*GammaUPz).s0 = read_imagef(Gamma_ztt_t1, sampler, coords).x;
    (*GammaUPz).s1 = read_imagef(Gamma_ztx_t1, sampler, coords).x;
    (*GammaUPz).s2 = read_imagef(Gamma_zty_t1, sampler, coords).x;
    (*GammaUPz).s3 = read_imagef(Gamma_ztz_t1, sampler, coords).x;
    /* GammaUPz.s4 = GammaUPz.s1; */
    (*GammaUPz).s5 = read_imagef(Gamma_zxx_t1, sampler, coords).x;
    (*GammaUPz).s6 = read_imagef(Gamma_zxy_t1, sampler, coords).x;
    (*GammaUPz).s7 = read_imagef(Gamma_zxz_t1, sampler, coords).x;
    /* GammaUPz.s8 = GammaUPz.s2; */
    /* GammaUPz.s9 = GammaUPz.s6; */
    (*GammaUPz).sa = read_imagef(Gamma_zyy_t1, sampler, coords).x;
    (*GammaUPz).sb = read_imagef(Gamma_zyz_t1, sampler, coords).x;
    /* GammaUPz.sc = GammaUPz.s3; */
    /* GammaUPz.sd = GammaUPz.s7; */
    /* GammaUPz.se = GammaUPz.sb; */
    (*GammaUPz).sf = read_imagef(Gamma_zzz_t1, sampler, coords).x;

    (*GammaUPz).s489 = (*GammaUPz).s126;
    (*GammaUPz).scde = (*GammaUPz).s37b;
  }else{
    (*GammaUPt).s0 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_ttt_t1, sampler, coords).x,
                                      read_imagef(Gamma_ttt_t2, sampler, coords).x);
    (*GammaUPt).s1 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_ttx_t1, sampler, coords).x,
                                      read_imagef(Gamma_ttx_t2, sampler, coords).x);
    (*GammaUPt).s2 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_tty_t1, sampler, coords).x,
                                      read_imagef(Gamma_tty_t2, sampler, coords).x);
    (*GammaUPt).s3 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_ttz_t1, sampler, coords).x,
                                      read_imagef(Gamma_ttz_t2, sampler, coords).x);
    /* GammaUPt.s4 = GammaUPt.s1; */
    (*GammaUPt).s5 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_txx_t1, sampler, coords).x,
                                      read_imagef(Gamma_txx_t2, sampler, coords).x);
    (*GammaUPt).s6 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_txy_t1, sampler, coords).x,
                                      read_imagef(Gamma_txy_t2, sampler, coords).x);
    (*GammaUPt).s7 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_txz_t1, sampler, coords).x,
                                      read_imagef(Gamma_txz_t2, sampler, coords).x);
    /* GammaUPt.s8 = GammaUPt.s2; */
    /* GammaUPt.s9 = GammaUPt.s6; */
    (*GammaUPt).sa = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_tyy_t1, sampler, coords).x,
                                      read_imagef(Gamma_tyy_t2, sampler, coords).x);
    (*GammaUPt).sb = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_tyz_t1, sampler, coords).x,
                                      read_imagef(Gamma_tyz_t2, sampler, coords).x);
    /* GammaUPt.sc = GammaUPt.s3; */
    /* GammaUPt.sd = GammaUPt.s7; */
    /* GammaUPt.se = GammaUPt.sb; */
    (*GammaUPt).sf = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_tzz_t1, sampler, coords).x,
                                      read_imagef(Gamma_tzz_t2, sampler, coords).x);

    (*GammaUPt).s489 = (*GammaUPt).s126;
    (*GammaUPt).scde = (*GammaUPt).s37b;


    (*GammaUPx).s0 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_xtt_t1, sampler, coords).x,
                                      read_imagef(Gamma_xtt_t2, sampler, coords).x);
    (*GammaUPx).s1 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_xtx_t1, sampler, coords).x,
                                      read_imagef(Gamma_xtx_t2, sampler, coords).x);
    (*GammaUPx).s2 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_xty_t1, sampler, coords).x,
                                      read_imagef(Gamma_xty_t2, sampler, coords).x);
    (*GammaUPx).s3 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_xtz_t1, sampler, coords).x,
                                      read_imagef(Gamma_xtz_t2, sampler, coords).x);
    /* GammaUPx.s4 = GammaUPx.s1; */
    (*GammaUPx).s5 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_xxx_t1, sampler, coords).x,
                                      read_imagef(Gamma_xxx_t2, sampler, coords).x);
    (*GammaUPx).s6 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_xxy_t1, sampler, coords).x,
                                      read_imagef(Gamma_xxy_t2, sampler, coords).x);
    (*GammaUPx).s7 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_xxz_t1, sampler, coords).x,
                                      read_imagef(Gamma_xxz_t2, sampler, coords).x);
    /* GammaUPx.s8 = GammaUPx.s2; */
    /* GammaUPx.s9 = GammaUPx.s6; */
    (*GammaUPx).sa = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_xyy_t1, sampler, coords).x,
                                      read_imagef(Gamma_xyy_t2, sampler, coords).x);
    (*GammaUPx).sb = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_xyz_t1, sampler, coords).x,
                                      read_imagef(Gamma_xyz_t2, sampler, coords).x);
    /* GammaUPx.sc = GammaUPx.s3; */
    /* GammaUPx.sd = GammaUPx.s7; */
    /* GammaUPx.se = GammaUPx.sb; */
    (*GammaUPx).sf = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_xzz_t1, sampler, coords).x,
                                      read_imagef(Gamma_xzz_t2, sampler, coords).x);

    (*GammaUPx).s489 = (*GammaUPx).s126;
    (*GammaUPx).scde = (*GammaUPx).s37b;


    (*GammaUPy).s0 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_ytt_t1, sampler, coords).x,
                                      read_imagef(Gamma_ytt_t2, sampler, coords).x);
    (*GammaUPy).s1 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_ytx_t1, sampler, coords).x,
                                      read_imagef(Gamma_ytx_t2, sampler, coords).x);
    (*GammaUPy).s2 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_yty_t1, sampler, coords).x,
                                      read_imagef(Gamma_yty_t2, sampler, coords).x);
    (*GammaUPy).s3 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_ytz_t1, sampler, coords).x,
                                      read_imagef(Gamma_ytz_t2, sampler, coords).x);
    /* GammaUPy.s4 = GammaUPy.s1; */
    (*GammaUPy).s5 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_yxx_t1, sampler, coords).x,
                                      read_imagef(Gamma_yxx_t2, sampler, coords).x);
    (*GammaUPy).s6 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_yxy_t1, sampler, coords).x,
                                      read_imagef(Gamma_yxy_t2, sampler, coords).x);
    (*GammaUPy).s7 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_yxz_t1, sampler, coords).x,
                                      read_imagef(Gamma_yxz_t2, sampler, coords).x);
    /* GammaUPy.s8 = GammaUPy.s2; */
    /* GammaUPy.s9 = GammaUPy.s6; */
    (*GammaUPy).sa = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_yyy_t1, sampler, coords).x,
                                      read_imagef(Gamma_yyy_t2, sampler, coords).x);
    (*GammaUPy).sb = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_yyz_t1, sampler, coords).x,
                                      read_imagef(Gamma_yyz_t2, sampler, coords).x);
    /* GammaUPy.sc = GammaUPy.s3; */
    /* GammaUPy.sd = GammaUPy.s7; */
    /* GammaUPy.se = GammaUPy.sb; */
    (*GammaUPy).sf = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_yzz_t1, sampler, coords).x,
                                      read_imagef(Gamma_yzz_t2, sampler, coords).x);

    (*GammaUPy).s489 = (*GammaUPy).s126;
    (*GammaUPy).scde = (*GammaUPy).s37b;


    (*GammaUPz).s0 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_ztt_t1, sampler, coords).x,
                                      read_imagef(Gamma_ztt_t2, sampler, coords).x);
    (*GammaUPz).s1 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_ztx_t1, sampler, coords).x,
                                      read_imagef(Gamma_ztx_t2, sampler, coords).x);
    (*GammaUPz).s2 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_zty_t1, sampler, coords).x,
                                      read_imagef(Gamma_zty_t2, sampler, coords).x);
    (*GammaUPz).s3 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_ztz_t1, sampler, coords).x,
                                      read_imagef(Gamma_ztz_t2, sampler, coords).x);
    /* GammaUPz.s4 = GammaUPz.s1; */
    (*GammaUPz).s5 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_zxx_t1, sampler, coords).x,
                                      read_imagef(Gamma_zxx_t2, sampler, coords).x);
    (*GammaUPz).s6 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_zxy_t1, sampler, coords).x,
                                      read_imagef(Gamma_zxy_t2, sampler, coords).x);
    (*GammaUPz).s7 = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_zxz_t1, sampler, coords).x,
                                      read_imagef(Gamma_zxz_t2, sampler, coords).x);
    /* GammaUPz.s8 = GammaUPz.s2; */
    /* GammaUPz.s9 = GammaUPz.s6; */
    (*GammaUPz).sa = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_zyy_t1, sampler, coords).x,
                                      read_imagef(Gamma_zyy_t2, sampler, coords).x);
    (*GammaUPz).sb = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_zyz_t1, sampler, coords).x,
                                      read_imagef(Gamma_zyz_t2, sampler, coords).x);
    /* GammaUPz.sc = GammaUPz.s3; */
    /* GammaUPz.sd = GammaUPz.s7; */
    /* GammaUPz.se = GammaUPz.sb; */
    (*GammaUPz).sf = time_interpolate(t, t1, t2,
                                      read_imagef(Gamma_zzz_t1, sampler, coords).x,
                                      read_imagef(Gamma_zzz_t2, sampler, coords).x);

    (*GammaUPz).s489 = (*GammaUPz).s126;
    (*GammaUPz).scde = (*GammaUPz).s37b;

  }
}



struct gr
gr_rhs(struct gr g, SPACETIME_PROTOTYPE_ARGS)
{
  real4 q = g.q;
  real4 u = g.u;

  /* If all the inequalities are satisfied, then inside_boundary is 1, if one
   * or more are not, then it will be 0. */
  int inside_boundary = is_q_inside_boundary(q, bounding_box);

  if (!inside_boundary){
    /* We are outside the boundary, we use Kerr-Schild with spin = 0 */

    /* TODO: Be more efficient here are re-write the equations taking advantage
       of the 0 spin */

    real  f,  dx_f,  dy_f,  dz_f;
    real  lx, dx_lx, dy_lx, dz_lx;
    real  ly, dx_ly, dy_ly, dz_ly;
    real  lz, dx_lz, dy_lz, dz_lz;

    real  hDxu, hDyu, hDzu;
    real4 uD;
    real  tmp;

    real a_spin = K(0.0);

    {
      real dx_r, dy_r, dz_r;
      real r, ir, iss;
      {
        real aa = a_spin * a_spin;
        real rr, tmp2;
        {
          real zz = q.s3 * q.s3;
          real dd;
          {
            real kk = K(0.5) * (q.s1 * q.s1 + q.s2 * q.s2 + zz - aa);
            dd = sqrt(kk * kk + aa * zz);
            rr = dd + kk;
          }
          r  = sqrt(rr);
          ir = K(1.0) / r;
          {
            real ss = rr + aa;
            iss  = K(1.0) / ss;
            tmp  = K(0.5) / (r * dd);
            dz_r = tmp * ss * q.s3;
            tmp *= rr;
          }
          dy_r = tmp * q.s2;
          dx_r = tmp * q.s1;
          tmp  = K(2.0) / (rr + aa * zz / rr);
        }
        tmp2 = K(3.0) - K(2.0) * rr * tmp;
        f    = tmp *  r;
        dx_f = tmp *  dx_r * tmp2;
        dy_f = tmp *  dy_r * tmp2;
        dz_f = tmp * (dz_r * tmp2 - tmp * aa * q.s3 * ir);
      } /* 48 (-8) FLOPs; estimated FLoating-point OPerations, the number
           in the parentheses is (the negative of) the number of FMA */
      {
        real m2r  = K(-2.0) * r;
        real issr = iss     * r;
        real issa = iss     * a_spin;

        lx    = iss * (q.s1 * r + q.s2 * a_spin);
        tmp   = iss * (q.s1 + m2r * lx);
        dx_lx = tmp * dx_r + issr;
        dy_lx = tmp * dy_r + issa;
        dz_lx = tmp * dz_r;

        ly    = iss * (q.s2 * r - q.s1 * a_spin);
        tmp   = iss * (q.s2 + m2r * ly);
        dx_ly = tmp * dx_r - issa;
        dy_ly = tmp * dy_r + issr;
        dz_ly = tmp * dz_r;

        lz    = q.s3 * ir;
        tmp   = -lz * ir;
        dx_lz = tmp * dx_r;
        dy_lz = tmp * dy_r;
        dz_lz = tmp * dz_r + ir;
      } /* 35 (-9) FLOPs */
    }

    {
      real  flu;
      real4 Dx, Dy, Dz;
      {
        real lu = u.s0 + lx * u.s1 + ly * u.s2 + lz * u.s3;
        flu   = f * lu;
        Dx.s0 = dx_f * lu + f * (dx_lx * u.s1 + dx_ly * u.s2 + dx_lz * u.s3);
        Dy.s0 = dy_f * lu + f * (dy_lx * u.s1 + dy_ly * u.s2 + dy_lz * u.s3);
        Dz.s0 = dz_f * lu + f * (dz_lx * u.s1 + dz_ly * u.s2 + dz_lz * u.s3); /* 31 (-12) FLOPs */
      }
      Dx.s1 = Dx.s0 * lx + flu * dx_lx;
      Dx.s2 = Dx.s0 * ly + flu * dx_ly;
      Dx.s3 = Dx.s0 * lz + flu * dx_lz; /* 9 (-3) FLOPs */

      Dy.s1 = Dy.s0 * lx + flu * dy_lx;
      Dy.s2 = Dy.s0 * ly + flu * dy_ly;
      Dy.s3 = Dy.s0 * lz + flu * dy_lz; /* 9 (-3) FLOPs */

      Dz.s1 = Dz.s0 * lx + flu * dz_lx;
      Dz.s2 = Dz.s0 * ly + flu * dz_ly;
      Dz.s3 = Dz.s0 * lz + flu * dz_lz; /* 9 (-3) FLOPs */

      hDxu = K(0.5) * dot(Dx, u);
      hDyu = K(0.5) * dot(Dy, u);
      hDzu = K(0.5) * dot(Dz, u); /* 24 (-9) FLOPs */

      uD  = u.s1 * Dx + u.s2 * Dy + u.s3 * Dz; /* 20 (-8) FLOPs */

      tmp = f * (-uD.s0 + lx * (uD.s1 - hDxu) + ly * (uD.s2 - hDyu) + lz * (uD.s3 - hDzu)); /* 10 (-3) FLOPs */
    }

    {
      real4 a = {
      uD.s0 -      tmp,
      hDxu - uD.s1 + lx * tmp,
      hDyu - uD.s2 + ly * tmp,
      hDzu - uD.s3 + lz * tmp
    }; /* 10 (-3) FLOPs */

      return (struct gr){u, a - a.s0 * u};
    }

  }else{

    /* Here we are inside the boundary */

    real16 GammaUPt, GammaUPx, GammaUPy, GammaUPz;

    interp_gammas(q, &GammaUPt, &GammaUPx, &GammaUPy, &GammaUPz, SPACETIME_ARGS);

    real4 rhs = {-dot(u, matrix_vector_product(GammaUPt, u)),
    -dot(u, matrix_vector_product(GammaUPx, u)),
    -dot(u, matrix_vector_product(GammaUPy, u)),
    -dot(u, matrix_vector_product(GammaUPz, u))};

    return (struct gr){u, rhs - rhs.s0 * u};
  }
}
