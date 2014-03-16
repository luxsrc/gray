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
#ifdef EBUG
#  define DEBUG 1
#  undef   EBUG
#endif

#ifdef OUBLE
#  if OUBLE == 1
#    define DOUBLE 1
#  else
#    define SINGLE 1
#  endif
#  undef OUBLE
#else
#  define MIXED 1
#endif

#ifdef ISABLE_GL
#  undef ISABLE_GL
#else
#  define ENABLE_GL 1
#  define INTEROPERABLE 1 // so class Data uses OpenGL buffer data
#endif

#ifdef ISABLE_PRIME
#  undef ISABLE_PRIME
#else
#  define ENABLE_PRIME 1
#endif

#ifdef ISABLE_LEAP
#  undef ISABLE_LEAP
#else
#  define ENABLE_LEAP 1
#endif

// Include system headers
#include <cuda_runtime_api.h> // C-style CUDA runtime API
#ifdef ENABLE_GL
#  ifdef __APPLE__
#    include <GLUT/glut.h>
#  else
#    include <GL/glut.h>
#  endif
#  include <GLFW/glfw3.h>
#endif

// Typedef real to make the source code precision independent
#if defined(DOUBLE)
  typedef double  real;
  typedef double xreal;
# define GL_REAL  GL_DOUBLE
# define GL_XREAL GL_DOUBLE
#elif defined(MIXED)
  typedef float   real;
  typedef double xreal;
# define GL_REAL  GL_FLOAT
# define GL_XREAL GL_DOUBLE
#elif defined(SINGLE)
  typedef float   real;
  typedef float  xreal;
# define GL_REAL  GL_FLOAT
# define GL_XREAL GL_FLOAT
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
#ifdef ENABLE_GL
  extern GLFWwindow *window;
  extern int   width, height;
  extern float ratio, ax, ly, az, a_spin;
  extern int   shader;
  extern bool  draw_body;
#endif
  extern size_t bsz;
}

// Basic function prototypes
extern void print(const char *, ...);
extern void error(const char *, ...);
#ifdef DEBUG
#  define debug(...) print("[DEBUG] " __VA_ARGS__)
#else
#  define debug(...) // do nothing
#endif
extern void optimize(int);

// NiTE+OpenNI or Leap Motion related functions for natural interactions
#if defined(ENABLE_PRIME) || defined(ENABLE_LEAP)
extern void sense();
#endif

// OpenGL/GLUT functions
#ifdef ENABLE_GL
extern void setup(int, char **);
extern void vis(GLuint, size_t);
#endif

// GRay specific functions
extern double evolve(Data &, double);
extern void   init  (Data &);
extern int    solve (Data &);

// Dirty wrapper functions that allow us to configure the CUDA kernels
extern bool init_config(const char *);
extern bool init_config(char, real  );
extern bool prob_config(const char *);
extern bool prob_config(char, real  );

#endif // GRAY_H
