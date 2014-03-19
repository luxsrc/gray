// Copyright (C) 2014 Chi-kwan Chan
// Copyright (C) 2014 Steward Observatory
//
// This file is part of GRay.
//
// GRay is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// GRay is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
// or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
// License for more details.
//
// You should have received a copy of the GNU General Public License
// along with GRay.  If not, see <http://www.gnu.org/licenses/>.

#ifndef MAKE_H
#define MAKE_H // Translate macros that are passed from the Makefile

#ifdef     EBUG
#  undef   EBUG
#  define DEBUG 1
#endif

#ifdef   OUBLE
#  if    OUBLE == 1
#    define DOUBLE  1
#    define GL_REAL GL_DOUBLE
#  else
#    define SINGLE  1
#    define GL_REAL GL_FLOAT
#  endif
#  undef OUBLE
#else
#  define MIXED   1
#  define GL_REAL GL_FLOAT
#endif

#ifdef    ISABLE_PROF
#  undef  ISABLE_PROF
#else
#  define ENABLE_PROF
#endif

#ifdef    ISABLE_GL
#  undef  ISABLE_GL
#else
#  define ENABLE_GL 1
#  ifdef    ISABLE_PRIME
#    undef  ISABLE_PRIME
#  else
#    define ENABLE_PRIME 1
#  endif
#  ifdef    ISABLE_LEAP
#    undef  ISABLE_LEAP
#  else
#    define ENABLE_LEAP 1
#  endif
#endif

#endif // MAKE_H
