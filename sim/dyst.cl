/* Automatically generated, do not edit */
inline real GRAY_SQUARE (real x) { return x*x; };
inline real GRAY_CUBE (real x) { return x*x*x; };
inline real GRAY_FOUR (real x) { return x*x*x*x; };
inline real GRAY_SQRT (real x) { return sqrt(x); };
inline real GRAY_SQRT_CUBE (real x) { return sqrt(x*x*x); };

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
getuu(real4 q, /**< Spacetime event "location" */
      real4 u) /**< The vector being squared   */
{
  real t = q.s0;
  real x = q.s1;
  real y = q.s2;
  real z = q.s3;

  return 0;

  /*   return (real4){dot(u, matrix_vector_product(g, u)), */
  /* fabs(dot(g.s0123, u) * u.s0) + */
  /* fabs(dot(g.s4567, u) * u.s1) + */
  /* fabs(dot(g.s89ab, u) * u.s2) + */
  /* fabs(dot(g.scdef, u) * u.s3), */
  /* K(0.0), K(0.0)}; */
}

real8
icond(real r_obs, /**< Distance of the observer from the black hole */
      real i_obs, /**< Inclination angle of the observer in degrees */
      real j_obs, /**< Azimuthal   angle of the observer in degrees */
      real alpha, /**< One of the local Cartesian coordinates       */
      real beta)  /**< The other  local Cartesian coordinate        */
{

  real  deg2rad = K(3.14159265358979323846264338327950288) / K(180.0);
  real  ci, si  = sincos(deg2rad * i_obs, &ci);
  real  cj, sj  = sincos(deg2rad * j_obs, &cj);

  real  R0 = r_obs * si - beta  * ci;
  real  z  = r_obs * ci + beta  * si;
  real  y  = R0    * sj - alpha * cj;
  real  x  = R0    * cj + alpha * sj;

  real4 q = (real4){K(0.0), x, y, z};
  real4 u = (real4){K(1.0), si * cj, si * sj, ci};

  return (real8){q, u};
}

real4 physical_coords_to_normalized_coords(real4 xyz, real8 bounding_box){

  const real x = xyz.s0;
  const real y = xyz.s1;
  const real z = xyz.s2;
  const real xmin = bounding_box.s0;
  const real xmax = bounding_box.s4;
  const real ymin = bounding_box.s1;
  const real ymax = bounding_box.s5;
  const real zmin = bounding_box.s2;
  const real zmax = bounding_box.s6;

  return (real4){(x - xmin)/(xmax - xmin),
                 (y - ymin)/(ymax - ymin),
                 (z - zmin)/(zmax - zmin),
                  0};
}

inline real interpolate(real4 xyz, real8 bounding_box, __read_only image3d_t var){

  sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

  return read_imagef(var,
                     sampler,
                     physical_coords_to_normalized_coords(xyz, bounding_box)).x;
}

real8
rhs(real8 s,
    const    real8 bounding_box, /**< Max coordinates of the grid    */
    __read_only image3d_t Gamma_ttt,
    __read_only image3d_t Gamma_ttx,
    __read_only image3d_t Gamma_tty,
    __read_only image3d_t Gamma_ttz,
    __read_only image3d_t Gamma_txx,
    __read_only image3d_t Gamma_txy,
    __read_only image3d_t Gamma_txz,
    __read_only image3d_t Gamma_tyy,
    __read_only image3d_t Gamma_tyz,
    __read_only image3d_t Gamma_tzz,
    __read_only image3d_t Gamma_xtt,
    __read_only image3d_t Gamma_xtx,
    __read_only image3d_t Gamma_xty,
    __read_only image3d_t Gamma_xtz,
    __read_only image3d_t Gamma_xxx,
    __read_only image3d_t Gamma_xxy,
    __read_only image3d_t Gamma_xxz,
    __read_only image3d_t Gamma_xyy,
    __read_only image3d_t Gamma_xyz,
    __read_only image3d_t Gamma_xzz,
    __read_only image3d_t Gamma_ytt,
    __read_only image3d_t Gamma_ytx,
    __read_only image3d_t Gamma_yty,
    __read_only image3d_t Gamma_ytz,
    __read_only image3d_t Gamma_yxx,
    __read_only image3d_t Gamma_yxy,
    __read_only image3d_t Gamma_yxz,
    __read_only image3d_t Gamma_yyy,
    __read_only image3d_t Gamma_yyz,
    __read_only image3d_t Gamma_yzz,
    __read_only image3d_t Gamma_ztt,
    __read_only image3d_t Gamma_ztx,
    __read_only image3d_t Gamma_zty,
    __read_only image3d_t Gamma_ztz,
    __read_only image3d_t Gamma_zxx,
    __read_only image3d_t Gamma_zxy,
    __read_only image3d_t Gamma_zxz,
    __read_only image3d_t Gamma_zyy,
    __read_only image3d_t Gamma_zyz,
    __read_only image3d_t Gamma_zzz)
{
  real4 q = s.s0123;
  real4 u = s.s4567;

  real t = q.s0;
  real x = q.s1;
  real y = q.s2;
  real z = q.s3;

  real4 xyz = {x, y, z, 0};

  real16 GammaUPt, GammaUPx, GammaUPy, GammaUPz;

  /* We compute the commented ones in one shot */
  GammaUPt.s0 = interpolate(xyz, bounding_box, Gamma_ttt);
  GammaUPt.s1 = interpolate(xyz, bounding_box, Gamma_ttx);
  GammaUPt.s2 = interpolate(xyz, bounding_box, Gamma_tty);
  GammaUPt.s3 = interpolate(xyz, bounding_box, Gamma_ttz);
  /* GammaUPt.s4 = GammaUPt.s0; */
  GammaUPt.s5 = interpolate(xyz, bounding_box, Gamma_txx);
  GammaUPt.s6 = interpolate(xyz, bounding_box, Gamma_txy);
  GammaUPt.s7 = interpolate(xyz, bounding_box, Gamma_txz);
  /* GammaUPt.s8 = GammaUPt.s2; */
  /* GammaUPt.s9 = GammaUPt.s6; */
  GammaUPt.sa = interpolate(xyz, bounding_box, Gamma_tyy);
  GammaUPt.sb = interpolate(xyz, bounding_box, Gamma_tyz);
  /* GammaUPt.sc = GammaUPt.s3; */
  /* GammaUPt.sd = GammaUPt.s7; */
  /* GammaUPt.se = GammaUPt.sb; */
  GammaUPt.sf = interpolate(xyz, bounding_box, Gamma_tzz);

  GammaUPt.s489 = GammaUPt.s026;
  GammaUPt.scde = GammaUPt.s37b;


  GammaUPx.s0 = interpolate(xyz, bounding_box, Gamma_xtt);
  GammaUPx.s1 = interpolate(xyz, bounding_box, Gamma_xtx);
  GammaUPx.s2 = interpolate(xyz, bounding_box, Gamma_xty);
  GammaUPx.s3 = interpolate(xyz, bounding_box, Gamma_xtz);
  GammaUPx.s5 = interpolate(xyz, bounding_box, Gamma_xxx);
  GammaUPx.s6 = interpolate(xyz, bounding_box, Gamma_xxy);
  GammaUPx.s7 = interpolate(xyz, bounding_box, Gamma_xxz);
  GammaUPx.sa = interpolate(xyz, bounding_box, Gamma_xyy);
  GammaUPx.sb = interpolate(xyz, bounding_box, Gamma_xyz);
  GammaUPx.sf = interpolate(xyz, bounding_box, Gamma_xzz);

  GammaUPx.s489 = GammaUPx.s026;
  GammaUPx.scde = GammaUPx.s37b;


  GammaUPy.s0 = interpolate(xyz, bounding_box, Gamma_ytt);
  GammaUPy.s1 = interpolate(xyz, bounding_box, Gamma_ytx);
  GammaUPy.s2 = interpolate(xyz, bounding_box, Gamma_yty);
  GammaUPy.s3 = interpolate(xyz, bounding_box, Gamma_ytz);
  GammaUPy.s5 = interpolate(xyz, bounding_box, Gamma_yxx);
  GammaUPy.s6 = interpolate(xyz, bounding_box, Gamma_yxy);
  GammaUPy.s7 = interpolate(xyz, bounding_box, Gamma_yxz);
  GammaUPy.sa = interpolate(xyz, bounding_box, Gamma_yyy);
  GammaUPy.sb = interpolate(xyz, bounding_box, Gamma_yyz);
  GammaUPy.sf = interpolate(xyz, bounding_box, Gamma_yzz);

  GammaUPy.s489 = GammaUPy.s026;
  GammaUPy.scde = GammaUPy.s37b;


  GammaUPz.s0 = interpolate(xyz, bounding_box, Gamma_ztt);
  GammaUPz.s1 = interpolate(xyz, bounding_box, Gamma_ztx);
  GammaUPz.s2 = interpolate(xyz, bounding_box, Gamma_zty);
  GammaUPz.s3 = interpolate(xyz, bounding_box, Gamma_ztz);
  GammaUPz.s5 = interpolate(xyz, bounding_box, Gamma_zxx);
  GammaUPz.s6 = interpolate(xyz, bounding_box, Gamma_zxy);
  GammaUPz.s7 = interpolate(xyz, bounding_box, Gamma_zxz);
  GammaUPz.sa = interpolate(xyz, bounding_box, Gamma_zyy);
  GammaUPz.sb = interpolate(xyz, bounding_box, Gamma_zyz);
  GammaUPz.sf = interpolate(xyz, bounding_box, Gamma_zzz);

  GammaUPz.s489 = GammaUPz.s026;
  GammaUPz.scde = GammaUPz.s37b;


  real4 rhs = {-dot(u, matrix_vector_product(GammaUPt, u)),
               -dot(u, matrix_vector_product(GammaUPx, u)),
               -dot(u, matrix_vector_product(GammaUPy, u)),
               -dot(u, matrix_vector_product(GammaUPz, u))};

  return (real8){u, rhs.s0, rhs.s1, rhs.s2, rhs.s3};
}
