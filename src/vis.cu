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

#include "geode.h"

#ifndef DISABLE_GL

#include <cuda_gl_interop.h> // OpenGL interoperability runtime API

typedef struct {
  float x, y, z;
  float r, g, b;
} Point;

static GLuint vbo = 0; // OpenGL Vertex Buffer Object
static struct cudaGraphicsResource *res = NULL;
static unsigned int shader = 0;

static float ax = 270, az = 90;
static float ly =-50;

static double dt_stored = 0.0;
static int last_x = 0, last_y = 0;
static int left   = 0, right  = 0;

static int sprites = 0;

#define GL_VERTEX_PROGRAM_POINT_SIZE_NV 0x8642

static const char vertex_shader[] =
  "void main()                                                            \n"
  "{                                                                      \n"
  "  vec4 vert = gl_Vertex;                                               \n"
  "  vert.w    = 1.0;                                                     \n"
  "  vec3 pos_eye = vec3(gl_ModelViewMatrix * vert);                      \n"
  "  gl_PointSize = max(1.0, 500.0 * gl_Point.size / (1.0 - pos_eye.z));  \n"
  "  gl_TexCoord[0] = gl_MultiTexCoord0;                                  \n"
  "  gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vert;       \n"
  "  gl_FrontColor = gl_Color;                                            \n"
  "  gl_FrontSecondaryColor = gl_SecondaryColor;                          \n"
  "}                                                                      \n";

static const char pixel_shader[] =
  "uniform sampler2D splatTexture;                                        \n"
  "void main()                                                            \n"
  "{                                                                      \n"
  "  vec4 color   = (0.6 + 0.4 * gl_Color)                                \n"
  "               * texture2D(splatTexture, gl_TexCoord[0].st);           \n"
  "  gl_FragColor = color * gl_SecondaryColor;                            \n"
  "}                                                                      \n";

static unsigned compile(const char *src, GLenum type)
{
  unsigned s = glCreateShader(type);
  glShaderSource(s, 1, &src, 0);
  glCompileShader(s);
  return s;
}

static unsigned char *mkimg(int n)
{
  unsigned char *img = new unsigned char[4 * n * n];

  for(int h = 0, i = 0; i < n; ++i) {
    double x  = 2 * (i + 0.5) / n - 1;
    double x2 = x * x;
    for(int j = 0; j < n; ++j, h += 4) {
      double y  = 2 * (j + 0.5) / n - 1;
      double r2 = x2 + y * y;
      if(r2 > 1) r2 = 1;
      img[h] = img[h+1] = img[h+2] = img[h+3] =
        255 * ((2 * sqrt(r2) - 3) * r2 + 1);
    }
  }

  return img;
}

static void display(void)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();

  glRotatef(-90, 1, 0, 0);
  glTranslatef(0, -ly, 0);
  glRotatef(-(az- 90), 1, 0, 0);
  glRotatef(-(ax-270), 0, 0, 1);

  // Draw wire sphere, i.e., the "black hole"
  glColor3f(0.0, 1.0, 0.0);
#ifdef A_SPIN
  glutWireSphere(1.0 + sqrt(1.0 - A_SPIN * A_SPIN), 32, 16);
#else
  glutWireSphere(1.0, 32, 16);
#endif

  // Draw particles, i.e., photon locations
  if(shader && sprites) {
    glUseProgram(shader);
    glUniform1i(glGetUniformLocation(shader, "splatTexture"), 0);
  }

  glEnable(GL_POINT_SPRITE_ARB);
  glEnable(GL_BLEND);
  glDepthMask(GL_FALSE);
  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);
  {
    const size_t full = sizeof(Point);
    const size_t half = full / 2;

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(3, GL_FLOAT, full, 0);
    glColorPointer (3, GL_FLOAT, full, (char *)half);
    glDrawArrays(GL_POINTS, 0, global::n);
  }
  glDisableClientState(GL_COLOR_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);
  glDepthMask(GL_TRUE);
  glDisable(GL_BLEND);
  glDisable(GL_POINT_SPRITE_ARB);

  glUseProgram(0);

  glutSwapBuffers();
}

static void reshape(int w, int h)
{
  glViewport(0, 0, w, h);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(27.0, (double)w / h, 1.0, 2500.0);

  glMatrixMode(GL_MODELVIEW);
}

static void keyboard(unsigned char key, int x, int y)
{
  using namespace global;

  switch(key) {
  case 27 : // ESCAPE key
  case 'q':
  case 'Q':
    exit(0);
    break;
  case 's':
  case 'S':
    sprites = !sprites;
    break;
  case 'r':
  case 'R':
    if(dt_dump == 0.0)
      dt_stored *= -1; // fall through
    else {
      dt_dump *= -1;
      break;
    }
  case 'p':
  case 'P':
    double temp = dt_stored;
    dt_stored = dt_dump;
    dt_dump = temp;
    break;
  }
}

static void mouse(int b, int s, int x, int y)
{
  last_x = x;
  last_y = y;

  switch(b) {
  case GLUT_LEFT_BUTTON:
    left  = (s == GLUT_DOWN);
    break;
  case GLUT_RIGHT_BUTTON:
    right = (s == GLUT_DOWN);
    break;
  }
}

static void motion(int x, int y)
{
  int dx = x - last_x; last_x = x;
  int dy = y - last_y; last_y = y;

  if(right)
    ly -= 0.05 * dy;
  else if(left) {
    az -= 0.5 * dy;
    ax -= 0.5 * dx;
  }

  glutPostRedisplay();
}

static void setup(void)
{
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);

  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);

  shader = glCreateProgram();
  glAttachShader(shader, compile(vertex_shader, GL_VERTEX_SHADER));
  glAttachShader(shader, compile(pixel_shader, GL_FRAGMENT_SHADER));
  glLinkProgram(shader);

  unsigned int texture;
  glGenTextures(1, (GLuint *)&texture);
  glBindTexture(GL_TEXTURE_2D, texture);

  glTexParameteri(GL_TEXTURE_2D,
                  GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
  glTexParameteri(GL_TEXTURE_2D,
                  GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D,
                  GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  unsigned char *img = mkimg(64);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 64, 64, 0,
               GL_RGBA, GL_UNSIGNED_BYTE, img);

  glActiveTextureARB(GL_TEXTURE0_ARB);
  glBindTexture(GL_TEXTURE_2D, texture);
  glSecondaryColor3f(0.8f, 0.4f, 0.1f);

  glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE);

  glEnable(GL_DEPTH_TEST);
  glClearColor(0.0, 0.0, 0.0, 1.0);

  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(Point) * global::n, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0); // unbind all buffer

  cudaGraphicsGLRegisterBuffer(&res, vbo, cudaGraphicsMapFlagsWriteDiscard);
}

#include <map.cu>

static __global__ void kernel(Point *p, const State *s, size_t n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < n) p[i] = map(s[i]);
}

void vis(void)
{
  if(!vbo) setup();

  size_t size = 0;
  void  *head = NULL;

  cudaGraphicsMapResources(1, &res, 0);
  cudaGraphicsResourceGetMappedPointer(&head, &size, res);
  {
    using namespace global;

    const int bsz = 256;
    const int gsz = (n - 1) / bsz + 1;

    kernel<<<gsz, bsz>>>((Point *)head, s, n);
  }
  cudaGraphicsUnmapResources(1, &res, 0); // unmap resource

  cudaDeviceSynchronize();
}

#endif // !DISABLE_GL
