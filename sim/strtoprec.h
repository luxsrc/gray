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
#ifndef _STRTOPREC_H_
#define _STRTOPREC_H_

#include <lux/switch.h>

#define GRAY_FLOAT    ((unsigned)sizeof(float))
#define GRAY_DOUBLE   ((unsigned)sizeof(double))
#define GRAY_EXTENDED ((unsigned)sizeof(long double))

static inline unsigned
strtoprec(const char *str)
{
	SWITCH {
	CASE(str[0] == 's') return GRAY_FLOAT;
	CASE(str[0] == 'f') return GRAY_FLOAT;
	CASE(str[0] == 'd') return GRAY_DOUBLE;
	CASE(str[0] == 'l') return GRAY_EXTENDED;
	DEFAULT             return 0U;
	}
}

#endif /* _STRTOPREC_H */
