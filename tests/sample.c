/*
 * Copyright (C) 2016 Chi-kwan Chan
 * Copyright (C) 2016 Steward Observatory
 *
 * This file is part of lux.
 *
 * lux is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * lux is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
 * License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with lux.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <lux.h>
#include <lux/dynamic.h>
#include <lux/mangle.h>
#include <lux/pvector.h>
#include <lux/solver.h>

#define LUX_RAP_CASTING 1
#include "sample_rap.h"

static int
driver(Lux_spec *s, Lux_args *a)
{
	size_t i, j;
	for(i = 0; i < s->n1; ++i) {
		for(j = 0; j < s->n2; ++j) {
			size_t h = i * s->n2 + j;
			if(h < s->n)
				a->z[h] = a->x[h] + a->alpha * a->y[h];
		}
	}
	return 0;
}

Lux_solution *
LUX_MOD(Lux_problem *prob, unsigned flags)
{
	Lux_spec *spec1 = mkspec(prob, (prob->n+ 1-1)/ 1,  1);
	Lux_spec *spec2 = mkspec(prob, (prob->n+ 2-1)/ 2,  2);
	Lux_spec *spec3 = mkspec(prob, (prob->n+ 4-1)/ 4,  4);
	Lux_spec *spec4 = mkspec(prob, (prob->n+ 8-1)/ 8,  8);
	Lux_spec *spec5 = mkspec(prob, (prob->n+16-1)/16, 16);
	Lux_spec *spec6 = mkspec(prob, (prob->n+32-1)/32, 32);
	Lux_spec *spec7 = mkspec(prob, (prob->n+64-1)/64, 64);

	Lux_args *args  = mkargs(prob);

	return pvector(
		Lux_solution,
		{{{driver, spec1}, args}, {0, 0, prob->n, 0}},
		{{{driver, spec2}, args}, {0, 0, prob->n, 0}},
		{{{driver, spec3}, args}, {0, 0, prob->n, 0}},
		{{{driver, spec4}, args}, {0, 0, prob->n, 0}},
		{{{driver, spec5}, args}, {0, 0, prob->n, 0}},
		{{{driver, spec6}, args}, {0, 0, prob->n, 0}},
		{{{driver, spec7}, args}, {0, 0, prob->n, 0}}
	);

	(void)flags; /* silence unused variable warning */
}
