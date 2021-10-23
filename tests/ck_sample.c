/*
 * Copyright (C) 2021 Gabriele Bozzola and Chi-kwan Chan
 * Copyright (C) 2021 Steward Observatory
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
#include <lux.h>
#include <lux/estimate.h>
#include <lux/measure.h>
#include <lux/planner.h>
#include <lux/pvector.h>
#include <lux/solver.h>
#include <stdlib.h>
#include <stdio.h>

#define LUX_RAP_CASTING 1
#include "sample_rap.h"

static double test_alpha = 3.0;

static void
test_init(Lux_problem *p)
{
	int i;
	for(i = 0; i < (int)p->n; ++i) {
		p->x[i] = i;
		p->y[i] = i * 2.0;
	}
}

static int
test_check(Lux_problem *p)
{
	int failed = 0, i;
	for(i = 0; i < (int)p->n; ++i)
		if(p->z[i] != 7.0 * i)
			failed = 1;
	return failed;
}

int
main(int argc, char *argv[])
{
	int failed = 0;

	Lux_problem   prob;
	Lux_solver   *solve;
	Lux_solution *sols;
	size_t        i, n;

	double m_best = HUGE_VAL;
	size_t i_best = 0;

	prob.n     = 1024 * 1024;
	prob.alpha = test_alpha;
	prob.x     = malloc(sizeof(double) * prob.n);
	prob.y     = malloc(sizeof(double) * prob.n);
	prob.z     = malloc(sizeof(double) * prob.n);

	lux_setup(&argc, &argv);

	lux_print("1. Load solvers from the current directory into planner ... ");
	solve = lux_load("sample", NULL);
	lux_print("%p DONE\n", solve);

	lux_print("2. Solve the problem... ");
	sols = solve(&prob, LUX_PLAN_EXHAUSTIVE);
	n    = pgetn(sols, 0);
	lux_print("%p; %zu solutions DONE\n", sols, n);

	lux_print("3. Estimate performance for the solutions ...\n");
	for(i = 0; i < n; ++i) {
		double e = estimate(&sols[i].opcnt);
		lux_print("   * Solution %zu, estimated cost = %g\n", i, e);
	}
	lux_print("   DONE\n");

	lux_print("4. Measure performance for the solutions ...\n");
	for(i = 0; i < n; ++i) {
		Lux_task *t = mkluxbasetask(sols[i].task);
		double    m = measure(t);
		free(t);
		lux_print("   * Solution %zu, measured cost = %g\n", i, m);
		if(m_best > m) {
			m_best = m;
			i_best = i;
		}
	}
	lux_print("   DONE\n");

	lux_print("5. Run the optimal solutoin %zu ... ", i_best);
	{
		Lux_task *t;

		test_init(&prob);

		t = mkluxbasetask(sols[i_best].task);
		t->exec(t);
		free(t);

		failed = test_check(&prob);
	}
	if(failed) {
		lux_print("FAILED\n");
		lux_abort();
	} else
		lux_print("DONE\n");

	lux_print("6. Free the solutions ... ");
	for(i = 0; i < n; ++i)
		free(sols[i].task.algo.spec);
	free(sols[0].task.args);
	pfree(sols);
	lux_print("DONE\n");

	lux_print("7. Unload the solver ... ");
	lux_unload(solve);
	lux_print("DONE\n");

	free(prob.x);
	free(prob.y);
	free(prob.z);

	return failed;
}
