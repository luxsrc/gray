// Copyright (C) 2012 Chi-kwan Chan
// Copyright (C) 2012 Steward Observatory
//
// This file is part of geode.
//
// Geode is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Geode is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
// or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
// License for more details.
//
// You should have received a copy of the GNU General Public License
// along with geode.  If not, see <http://www.gnu.org/licenses/>.

#ifndef GEODE_H
#define GEODE_H

// Rename macros
#ifdef EBUG
#  define DEBUG
#  undef   EBUG
#endif
#ifdef ISABLE_GL
#  define DISABLE_GL
#  undef   ISABLE_GL
#endif
#ifdef OUBLE
#  define DOUBLE
#  undef   OUBLE
#endif
#ifdef UMP
#  define DUMP
#  undef   UMP
#endif

// Include system headers
#include <cuda_runtime_api.h> // C-style CUDA runtime API
#ifndef DISABLE_GL
#  ifdef __APPLE__
#    include <GLUT/glut.h>
#  else
#    include <GL/glut.h>
#  endif
#endif

// Typedef real to make the source code precision independent
#ifdef DOUBLE
  typedef double real;
# define GL_REAL GL_DOUBLE
#else
  typedef float real;
# define GL_REAL GL_FLOAT
#endif

// Include problem specific headers
#include <para.h>  // problem parameter
#include <state.h> // problem specific state structure
#define NVAR (sizeof(State) / sizeof(real))

// Include the Data class, which needs the State type
#include "data.h"

// Global variables
namespace global {
  extern double t, dt_dump;
}

// Basic function prototypes
extern void print(const char *, ...);
extern void error(const char *, ...);
#ifdef DEBUG
#  define debug(...) print("[DEBUG] " __VA_ARGS__)
#else
#  define debug(...) // do nothing
#endif

// OpenGL/GLUT functions
#ifndef DISABLE_GL
extern void mktexture(GLuint[1]);
extern void mkshaders(GLuint[2]);
extern int  getctrl();
extern void regctrl();
extern void vis(GLuint, size_t);
#endif

// Geode specific functions
extern void  dump  (Data &);
extern float evolve(Data &, double);
extern void  init  (Data &);
extern int   solve (Data &);

#endif // GEODE_H
