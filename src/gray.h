// Copyright (C) 2012,2013 Chi-kwan Chan
// Copyright (C) 2012,2013 Steward Observatory
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
#ifdef EBUG
#  define DEBUG
#  undef   EBUG
#endif
#ifdef ISABLE_GL
#  define DISABLE_GL
#  undef   ISABLE_GL
#endif
#ifdef ISABLE_NITE
#  define DISABLE_NITE
#  undef   ISABLE_NITE
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
#include <state.h> // problem specific state structure
#define NVAR (sizeof(State) / sizeof(real))

// Include the Data class, which needs the State type
#include "data.h"

// Global variables
namespace global {
  extern double t, dt_dump, dt_saved;
  extern const char *format;
#ifndef DISABLE_GL
  extern float ratio, ax, ly, az, a_spin;
#endif
}

// Basic function prototypes
extern void print(const char *, ...);
extern void error(const char *, ...);
#ifdef DEBUG
#  define debug(...) print("[DEBUG] " __VA_ARGS__)
#else
#  define debug(...) // do nothing
#endif

// NiTE and OpenNI related functions for natural interactions
#ifndef DISABLE_NITE
extern void sense();
extern void track();
#endif

// OpenGL/GLUT functions
#ifndef DISABLE_GL
extern void mktexture(GLuint[1]);
extern void mkshaders(GLuint[2]);
extern int  getctrl();
extern void regctrl();
extern void vis(GLuint, size_t);
#endif

// GRay specific functions
extern void   dump  (Data &);
extern double evolve(Data &, double);
extern void   init  (Data &);
extern int    solve (Data &);

// Dirty wrapper functions that allow us to configure the CUDA kernels
extern bool init_config(const char *);
extern bool init_config(char, real  );
extern bool prob_config(const char *);
extern bool prob_config(char, real  );

#endif // GRAY_H
