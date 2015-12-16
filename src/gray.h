// Copyright (C) 2012--2014 Chi-kwan Chan
// Copyright (C) 2012--2014 Steward Observatory
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

#ifndef GRAY_H
#define GRAY_H

// Rename macros
#include "make.h"

// Include system and optional headers
#include <cstdio>
#include <cuda_runtime_api.h> // C-style CUDA runtime API

#ifdef ENABLE_GL
#  include "optional/vis.h"
#endif

#define DELTA 256
#define LIMIT (DELTA * DELTA)

// Typedef real to make the source code precision independent
#if defined(DOUBLE)
  typedef double  real;
  typedef double xreal;
#elif defined(MIXED)
  typedef float   real;
  typedef double xreal;
#elif defined(SINGLE)
  typedef float   real;
  typedef float  xreal;
#endif

// Include problem specific headers and main data objects
#include <const.h> // problem specific constants
#include <state.h> // problem specific state structure

#include "data.h" // Data is an object that holds State
#include "para.h" // Para is an object that holds Const

#define NVAR (sizeof(State) / sizeof(real))

// Scheme based functions to estimate performance
namespace scheme {
  extern double flop();
  extern double rwsz();
}

// Basic function prototypes
extern void print(const char *, ...);
extern void error(const char *, ...);
#ifdef DEBUG
#  define debug(...) print("[DEBUG] " __VA_ARGS__)
#else
#  define debug(...) // do nothing
#endif
extern const char *match(const char *, const char *);

extern int help(const char *);

// Redefine math functions
#if defined(DOUBLE)
#  if N_NU > 3
#    warning In double precision model, GRay produces wrong answer if N_NU > 3
#  endif
#  define FABS(x)         fabs(x)
#  define MIN(x, y)       fmin(x, y)

#  define EXP(x)          exp(x)
#  define LOG(x)          log(x)
#  define POW(x, y)       pow(x, y)

#  define SQRT(x)         sqrt(x)
#  define CBRT(x)         cbrt(x)

#  define SIN(x)          sin(x)
#  define COS(x)          cos(x)
#  define ACOS(x)         acos(x)
#  define ATAN2(y, x)     atan2(y, x)
#  define SINCOS(x, s, c) sincos(x, s, c)
#elif defined(MIXED)
#  error Double or mixed precisions may result incorrect answers
#elif defined(SINGLE)
#  define FABS(x)         fabsf(x)
#  define MIN(x, y)       fminf(x, y)

#  define EXP(x)          __expf(x)
#  define LOG(x)          __logf(x)
#  define POW(x, y)       __powf(x, y)

#  define SQRT(x)         __fsqrt_rn(x)
#  define CBRT(x)         cbrtf(x)

#  define SIN(x)          __sinf(x)
#  define COS(x)          __cosf(x)
#  define ACOS(x)         acosf(x)
#  define ATAN2(y, x)     atan2f(y, x)
#  define SINCOS(x, s, c) __sincosf(x, s, c)
#endif

#endif // GRAY_H
