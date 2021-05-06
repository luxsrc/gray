/*
 * Copyright (C) 2016 Chi-kwan Chan
 * Copyright (C) 2016 Steward Observatory
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
 * along with GRay2.  If not, see <http://www.gnu.org/licenses/>.
 */

/** \file
 ** Preamble: useful OpenCL macros and functions
 **
 ** GRay2 uses OpenCL's just-in-time compilation feature to implement
 ** run-time configurable algorithms.  In this preamble we provide
 ** OpenCL macros and functions that help implementing the other parts
 ** of GRay2.
 **/

#define EPSILON 1e-32

/** Helper macros to write equations for vector of length n_vars **/
#define EACH(s) for(whole _e_ = 0; _e_ < n_chunk; ++_e_) E(s)
#define E(s) ((realE *)&(s))[_e_]

/** Turn an expression into a local variable that can be passed to function **/
#define X(x) ({ struct state _; EACH(_) = (x); _; })

/** Spacetime arguments for functions **/
#define SPACETIME_PROTOTYPE_ARGS \
const    real8 bounding_box, /**< Max coordinates of the grid    */ \
const    int4 num_points, /**< Number of points on the grid    */ \
__read_only image3d_t Gamma_ttt_t1, \
__read_only image3d_t Gamma_ttx_t1, \
__read_only image3d_t Gamma_tty_t1, \
__read_only image3d_t Gamma_ttz_t1, \
__read_only image3d_t Gamma_txx_t1, \
__read_only image3d_t Gamma_txy_t1, \
__read_only image3d_t Gamma_txz_t1, \
__read_only image3d_t Gamma_tyy_t1, \
__read_only image3d_t Gamma_tyz_t1, \
__read_only image3d_t Gamma_tzz_t1, \
__read_only image3d_t Gamma_xtt_t1, \
__read_only image3d_t Gamma_xtx_t1, \
__read_only image3d_t Gamma_xty_t1, \
__read_only image3d_t Gamma_xtz_t1, \
__read_only image3d_t Gamma_xxx_t1, \
__read_only image3d_t Gamma_xxy_t1, \
__read_only image3d_t Gamma_xxz_t1, \
__read_only image3d_t Gamma_xyy_t1, \
__read_only image3d_t Gamma_xyz_t1, \
__read_only image3d_t Gamma_xzz_t1, \
__read_only image3d_t Gamma_ytt_t1, \
__read_only image3d_t Gamma_ytx_t1, \
__read_only image3d_t Gamma_yty_t1, \
__read_only image3d_t Gamma_ytz_t1, \
__read_only image3d_t Gamma_yxx_t1, \
__read_only image3d_t Gamma_yxy_t1, \
__read_only image3d_t Gamma_yxz_t1, \
__read_only image3d_t Gamma_yyy_t1, \
__read_only image3d_t Gamma_yyz_t1, \
__read_only image3d_t Gamma_yzz_t1, \
__read_only image3d_t Gamma_ztt_t1, \
__read_only image3d_t Gamma_ztx_t1, \
__read_only image3d_t Gamma_zty_t1, \
__read_only image3d_t Gamma_ztz_t1, \
__read_only image3d_t Gamma_zxx_t1, \
__read_only image3d_t Gamma_zxy_t1, \
__read_only image3d_t Gamma_zxz_t1, \
__read_only image3d_t Gamma_zyy_t1, \
__read_only image3d_t Gamma_zyz_t1, \
__read_only image3d_t Gamma_zzz_t1, \
__read_only image3d_t g_tt_t1, \
__read_only image3d_t g_tx_t1, \
__read_only image3d_t g_ty_t1, \
__read_only image3d_t g_tz_t1, \
__read_only image3d_t g_xx_t1, \
__read_only image3d_t g_xy_t1, \
__read_only image3d_t g_xz_t1, \
__read_only image3d_t g_yy_t1, \
__read_only image3d_t g_yz_t1, \
__read_only image3d_t g_zz_t1, \
__read_only image3d_t rho_t1, \
__read_only image3d_t bt_t1, \
__read_only image3d_t bx_t1, \
__read_only image3d_t by_t1, \
__read_only image3d_t bz_t1, \
__read_only image3d_t Gamma_ttt_t2, \
__read_only image3d_t Gamma_ttx_t2, \
__read_only image3d_t Gamma_tty_t2, \
__read_only image3d_t Gamma_ttz_t2, \
__read_only image3d_t Gamma_txx_t2, \
__read_only image3d_t Gamma_txy_t2, \
__read_only image3d_t Gamma_txz_t2, \
__read_only image3d_t Gamma_tyy_t2, \
__read_only image3d_t Gamma_tyz_t2, \
__read_only image3d_t Gamma_tzz_t2, \
__read_only image3d_t Gamma_xtt_t2, \
__read_only image3d_t Gamma_xtx_t2, \
__read_only image3d_t Gamma_xty_t2, \
__read_only image3d_t Gamma_xtz_t2, \
__read_only image3d_t Gamma_xxx_t2, \
__read_only image3d_t Gamma_xxy_t2, \
__read_only image3d_t Gamma_xxz_t2, \
__read_only image3d_t Gamma_xyy_t2, \
__read_only image3d_t Gamma_xyz_t2, \
__read_only image3d_t Gamma_xzz_t2, \
__read_only image3d_t Gamma_ytt_t2, \
__read_only image3d_t Gamma_ytx_t2, \
__read_only image3d_t Gamma_yty_t2, \
__read_only image3d_t Gamma_ytz_t2, \
__read_only image3d_t Gamma_yxx_t2, \
__read_only image3d_t Gamma_yxy_t2, \
__read_only image3d_t Gamma_yxz_t2, \
__read_only image3d_t Gamma_yyy_t2, \
__read_only image3d_t Gamma_yyz_t2, \
__read_only image3d_t Gamma_yzz_t2, \
__read_only image3d_t Gamma_ztt_t2, \
__read_only image3d_t Gamma_ztx_t2, \
__read_only image3d_t Gamma_zty_t2, \
__read_only image3d_t Gamma_ztz_t2, \
__read_only image3d_t Gamma_zxx_t2, \
__read_only image3d_t Gamma_zxy_t2, \
__read_only image3d_t Gamma_zxz_t2, \
__read_only image3d_t Gamma_zyy_t2, \
__read_only image3d_t Gamma_zyz_t2, \
__read_only image3d_t Gamma_zzz_t2, \
__read_only image3d_t g_tt_t2, \
__read_only image3d_t g_tx_t2, \
__read_only image3d_t g_ty_t2, \
__read_only image3d_t g_tz_t2, \
__read_only image3d_t g_xx_t2, \
__read_only image3d_t g_xy_t2, \
__read_only image3d_t g_xz_t2, \
__read_only image3d_t g_yy_t2, \
__read_only image3d_t g_yz_t2, \
__read_only image3d_t g_zz_t2, \
__read_only image3d_t rho_t2, \
__read_only image3d_t bt_t2, \
__read_only image3d_t bx_t2, \
__read_only image3d_t by_t2, \
__read_only image3d_t bz_t2

#define SPACETIME_ARGS \
bounding_box, \
num_points,   \
Gamma_ttt_t1, \
Gamma_ttx_t1, \
Gamma_tty_t1, \
Gamma_ttz_t1, \
Gamma_txx_t1, \
Gamma_txy_t1, \
Gamma_txz_t1, \
Gamma_tyy_t1, \
Gamma_tyz_t1, \
Gamma_tzz_t1, \
Gamma_xtt_t1, \
Gamma_xtx_t1, \
Gamma_xty_t1, \
Gamma_xtz_t1, \
Gamma_xxx_t1, \
Gamma_xxy_t1, \
Gamma_xxz_t1, \
Gamma_xyy_t1, \
Gamma_xyz_t1, \
Gamma_xzz_t1, \
Gamma_ytt_t1, \
Gamma_ytx_t1, \
Gamma_yty_t1, \
Gamma_ytz_t1, \
Gamma_yxx_t1, \
Gamma_yxy_t1, \
Gamma_yxz_t1, \
Gamma_yyy_t1, \
Gamma_yyz_t1, \
Gamma_yzz_t1, \
Gamma_ztt_t1, \
Gamma_ztx_t1, \
Gamma_zty_t1, \
Gamma_ztz_t1, \
Gamma_zxx_t1, \
Gamma_zxy_t1, \
Gamma_zxz_t1, \
Gamma_zyy_t1, \
Gamma_zyz_t1, \
Gamma_zzz_t1, \
g_tt_t1, \
g_tx_t1, \
g_ty_t1, \
g_tz_t1, \
g_xx_t1, \
g_xy_t1, \
g_xz_t1, \
g_yy_t1, \
g_yz_t1, \
g_zz_t1, \
rho_t1, \
bt_t1, \
bx_t1, \
by_t1, \
bz_t1, \
Gamma_ttt_t2, \
Gamma_ttx_t2, \
Gamma_tty_t2, \
Gamma_ttz_t2, \
Gamma_txx_t2, \
Gamma_txy_t2, \
Gamma_txz_t2, \
Gamma_tyy_t2, \
Gamma_tyz_t2, \
Gamma_tzz_t2, \
Gamma_xtt_t2, \
Gamma_xtx_t2, \
Gamma_xty_t2, \
Gamma_xtz_t2, \
Gamma_xxx_t2, \
Gamma_xxy_t2, \
Gamma_xxz_t2, \
Gamma_xyy_t2, \
Gamma_xyz_t2, \
Gamma_xzz_t2, \
Gamma_ytt_t2, \
Gamma_ytx_t2, \
Gamma_yty_t2, \
Gamma_ytz_t2, \
Gamma_yxx_t2, \
Gamma_yxy_t2, \
Gamma_yxz_t2, \
Gamma_yyy_t2, \
Gamma_yyz_t2, \
Gamma_yzz_t2, \
Gamma_ztt_t2, \
Gamma_ztx_t2, \
Gamma_zty_t2, \
Gamma_ztz_t2, \
Gamma_zxx_t2, \
Gamma_zxy_t2, \
Gamma_zxz_t2, \
Gamma_zyy_t2, \
Gamma_zyz_t2, \
Gamma_zzz_t2, \
g_tt_t2, \
g_tx_t2, \
g_ty_t2, \
g_tz_t2, \
g_xx_t2, \
g_xy_t2, \
g_xz_t2, \
g_yy_t2, \
g_yz_t2, \
g_zz_t2, \
rho_t2, \
bt_t2, \
bx_t2, \
by_t2, \
bz_t2

#define HORIZON_PROTOTYPE_ARGS \
__global const int   *ah_valid_1, \
__global const real4 *ah_centr_1, \
__global const real  *ah_min_r_1, \
__global const real  *ah_max_r_1, \
__global const int   *ah_valid_2, \
__global const real4 *ah_centr_2, \
__global const real  *ah_min_r_2, \
__global const real  *ah_max_r_2, \
__global const int   *ah_valid_3, \
__global const real4 *ah_centr_3, \
__global const real  *ah_min_r_3, \
__global const real  *ah_max_r_3


#define HORIZON_ARGS \
ah_valid_1, \
ah_centr_1, \
ah_min_r_1, \
ah_max_r_1, \
ah_valid_2, \
ah_centr_2, \
ah_min_r_2, \
ah_max_r_2, \
ah_valid_3, \
ah_centr_3, \
ah_min_r_3, \
ah_max_r_3
