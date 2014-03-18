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
extern void pick(int);

#endif // GRAY_H
