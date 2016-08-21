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
	double4 q = s.s0123;
	double4 u = s.s4567;
	double4 a; /* "acceleration", not black hole spin */

	double f,  dx_f,  dy_f,  dz_f;
	double lx, dx_lx, dy_lx, dz_lx;
	double ly, dx_ly, dy_ly, dz_ly;
	double lz, dx_lz, dy_lz, dz_lz;
	double tmp;
	{
		double dx_r, dy_r, dz_r;
		double r, ir, iss;
		{
			double aa = a_spin * a_spin;
			double rr;
			{
				double zz = q.s3 * q.s3;
				double dd;
				{
					double kk = 0.5 * (q.s1 * q.s1 + q.s2 * q.s2 + zz - aa);
					dd = sqrt(kk * kk + aa * zz);
					rr = dd + kk;
				}
				r  = sqrt(rr);
				ir = 1.0 / r;
				{
					double ss = rr + aa;
					iss  = 1.0 / ss;
					tmp  = 0.5 / (r * dd);
					dz_r = tmp * ss * q.s3;
					tmp *= rr;
				}
				dy_r = tmp * q.s2;
				dx_r = tmp * q.s1;
				tmp  = 2.0 / (rr + aa * zz / rr);
			}
			f    = tmp *  r;
			dx_f = tmp *  dx_r * (3.0 - 2.0 * rr * tmp);
			dy_f = tmp *  dy_r * (3.0 - 2.0 * rr * tmp);
			dz_f = tmp * (dz_r * (3.0 - 2.0 * rr * tmp) - tmp * aa * q.s3 * ir);
		} /* 48 (-8) FLOPs; estimated FLoating-point OPerations, the number
		     in the parentheses is (the negative of) the number of FMA */
		{
			double m2r  = -2.0 * r;
			double issr =  iss * r;
			double issa =  iss * a_spin;

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
		double hDxu, hDyu, hDzu;
		double uDt, uDx, uDy, uDz;
		{
			double flu;
			double Dxt, Dyt, Dzt;
			{
				double lu = u.s0 + lx * u.s1 + ly * u.s2 + lz * u.s3;
				flu = f * lu;
				Dxt = dx_f * lu + f * (dx_lx * u.s1 + dx_ly * u.s2 + dx_lz * u.s3);
				Dyt = dy_f * lu + f * (dy_lx * u.s1 + dy_ly * u.s2 + dy_lz * u.s3);
				Dzt = dz_f * lu + f * (dz_lx * u.s1 + dz_ly * u.s2 + dz_lz * u.s3); /* 31 (-12) FLOPs */
			}
			double Dxx = Dxt * lx + flu * dx_lx;
			double Dxy = Dxt * ly + flu * dx_ly;
			double Dxz = Dxt * lz + flu * dx_lz; /* 9 (-3) FLOPs */

			double Dyx = Dyt * lx + flu * dy_lx;
			double Dyy = Dyt * ly + flu * dy_ly;
			double Dyz = Dyt * lz + flu * dy_lz; /* 9 (-3) FLOPs */

			double Dzx = Dzt * lx + flu * dz_lx;
			double Dzy = Dzt * ly + flu * dz_ly;
			double Dzz = Dzt * lz + flu * dz_lz; /* 9 (-3) FLOPs */

			hDxu = 0.5 * (Dxt * u.s0 + Dxx * u.s1 + Dxy * u.s2 + Dxz * u.s3);
			hDyu = 0.5 * (Dyt * u.s0 + Dyx * u.s1 + Dyy * u.s2 + Dyz * u.s3);
			hDzu = 0.5 * (Dzt * u.s0 + Dzx * u.s1 + Dzy * u.s2 + Dzz * u.s3); /* 24 (-9) FLOPs */

			uDt = u.s1 * Dxt + u.s2 * Dyt + u.s3 * Dzt;
			uDx = u.s1 * Dxx + u.s2 * Dyx + u.s3 * Dzx;
			uDy = u.s1 * Dxy + u.s2 * Dyy + u.s3 * Dzy;
			uDz = u.s1 * Dxz + u.s2 * Dyz + u.s3 * Dzz; /* 20 (-8) FLOPs */

			tmp = f * (-uDt + lx * (uDx - hDxu) + ly * (uDy - hDyu) + lz * (uDz - hDzu)); /* 10 (-3) FLOPs */
		}

		a.s0 =        uDt -      tmp;
		a.s1 = hDxu - uDx + lx * tmp;
		a.s2 = hDyu - uDy + ly * tmp;
		a.s3 = hDzu - uDz + lz * tmp; /* 10 (-3) FLOPs */
	}

	return (double8){u, a};
}
