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
double
getuu(double4 q, double4 u)
{
	double  aa = a_spin * a_spin;
	double  zz = q.s3 * q.s3;
	double  kk = 0.5 * (q.s1 * q.s1 + q.s2 * q.s2 + zz - aa);
	double  dd = sqrt(kk * kk + aa * zz);
	double  rr = dd + kk;
	double  r  = sqrt(rr);

	double  f  = 2.0 * rr * r / (rr * rr + aa * zz);
	double  lx = (r * q.s1 + a_spin * q.s2) / (rr + aa);
	double  ly = (r * q.s2 - a_spin * q.s1) / (rr + aa);
	double  lz = q.s3 / r;

	double4 gt = {-1. + f*1.*1.,      f*1.*lx,      f*1.*ly,      f*1.*lz};
	double4 gx = {      f*lx*1., 1. + f*lx*lx,      f*lx*ly,      f*lx*lz};
	double4 gy = {      f*ly*1.,      f*ly*lx, 1. + f*ly*ly,      f*ly*lz};
	double4 gz = {      f*lz*1.,      f*lz*lx,      f*lz*ly, 1. + f*lz*lz};

	return (dot(gt, u) * u.s0 +
	        dot(gx, u) * u.s1 +
	        dot(gy, u) * u.s2 +
	        dot(gz, u) * u.s3);
}

double8
icond(double r_obs, double i_obs, double j_obs, double alpha, double beta)
{
	double  ci, si = sincos(M_PI * i_obs / 180.0, &ci);
	double  cj, sj = sincos(M_PI * j_obs / 180.0, &cj);

	double  R = r_obs * si - beta  * ci; /* cylindrical radius */
	double  z = r_obs * ci + beta  * si;
	double  y = R     * sj - alpha * cj;
	double  x = R     * cj + alpha * sj;

	double4 q = (double4){0.0, x, y, z};
	double4 u = (double4){1.0, si * cj, si * sj, ci};

	double  aa = a_spin * a_spin;
	double  zz = q.s3 * q.s3;
	double  kk = 0.5 * (q.s1 * q.s1 + q.s2 * q.s2 + zz - aa);
	double  dd = sqrt(kk * kk + aa * zz);
	double  rr = dd + kk;
	double  r  = sqrt(rr);

	double  f  = 2.0 * rr * r / (rr * rr + aa * zz);
	double  lx = (r * q.s1 + a_spin * q.s2) / (rr + aa);
	double  ly = (r * q.s2 - a_spin * q.s1) / (rr + aa);
	double  lz = q.s3 / r;

	double4 gt = {-1. + f*1.*1.,      f*1.*lx,      f*1.*ly,      f*1.*lz};
	double4 gx = {      f*lx*1., 1. + f*lx*lx,      f*lx*ly,      f*lx*lz};
	double4 gy = {      f*ly*1.,      f*ly*lx, 1. + f*ly*ly,      f*ly*lz};
	double4 gz = {      f*lz*1.,      f*lz*lx,      f*lz*ly, 1. + f*lz*lz};

	double  A  =  gt.s0;
	double  B  =  dot(gt.s123, u.s123) * 2;
	double  C  = (dot(gx.s123, u.s123) * u.s1 +
	              dot(gy.s123, u.s123) * u.s2 +
	              dot(gz.s123, u.s123) * u.s3);

	u.s123 /= -(B + sqrt(B * B - 4 * A * C)) / (2 * A);

	return (double8){q, u};
}

double8
rhs(double8 s)
{
	/* TODO: actually implement the right hand side */
	return (double8){s.s4567, 0, 0, 0, 0};
}
