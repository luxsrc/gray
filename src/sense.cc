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

#include "gray.h"

#ifndef DISABLE_NITE
#include <cmath>
#include <NiTE.h>

static bool first = true;

static nite::UserTracker tracker;
static nite::Point3f head, neck, torso, upper[6], lower[6];

static bool left = false, right = false;

static void setup()
{
  print("Making sense...");

  if(nite::STATUS_OK != nite::NiTE::initialize())
    error("sense(): fail to initialize NiTE\n");

  if(nite::STATUS_OK != tracker.create())
    error("sense(): fail to create user tracker\n");

  first = false;

  print(" DONE\n");
}

static void cleanup()
{
  nite::NiTE::shutdown();
}

void sense()
{
  if(first && !atexit(cleanup)) setup();

  nite::UserTrackerFrameRef frame;
  if(nite::STATUS_OK == tracker.readFrame(&frame)) {
    const nite::Array<nite::UserData>& users = frame.getUsers();

    for(int i = 0; i < users.getSize(); ++i) {
      const nite::UserData& u = users[i];

      if(u.isNew())
        tracker.startSkeletonTracking(u.getId());
      else if(nite::SKELETON_TRACKED == u.getSkeleton().getState()) {
        const nite::Point3f h =
          u.getSkeleton().getJoint(nite::JOINT_HEAD).getPosition();
        print("head: (%g, %g, %g)\n", h.x, h.y, h.z);
      }
    }

    if(users.getSize() > 0) {
      const nite::Skeleton& s = users[0].getSkeleton();

      head     = s.getJoint(nite::JOINT_HEAD          ).getPosition();
      neck     = s.getJoint(nite::JOINT_NECK          ).getPosition();
      torso    = s.getJoint(nite::JOINT_TORSO         ).getPosition();
      upper[0] = s.getJoint(nite::JOINT_LEFT_HAND     ).getPosition();
      upper[1] = s.getJoint(nite::JOINT_LEFT_ELBOW    ).getPosition();
      upper[2] = s.getJoint(nite::JOINT_LEFT_SHOULDER ).getPosition();
      upper[3] = s.getJoint(nite::JOINT_RIGHT_SHOULDER).getPosition();
      upper[4] = s.getJoint(nite::JOINT_RIGHT_ELBOW   ).getPosition();
      upper[5] = s.getJoint(nite::JOINT_RIGHT_HAND    ).getPosition();
      lower[0] = s.getJoint(nite::JOINT_LEFT_FOOT     ).getPosition();
      lower[1] = s.getJoint(nite::JOINT_LEFT_KNEE     ).getPosition();
      lower[2] = s.getJoint(nite::JOINT_LEFT_HIP      ).getPosition();
      lower[3] = s.getJoint(nite::JOINT_RIGHT_HIP     ).getPosition();
      lower[4] = s.getJoint(nite::JOINT_RIGHT_KNEE    ).getPosition();
      lower[5] = s.getJoint(nite::JOINT_RIGHT_FOOT    ).getPosition();

      const nite::Point3f u(neck.x - torso.x,
                            neck.y - torso.y,
                            neck.z - torso.z);
      const nite::Point3f l(upper[0].x - upper[1].x,
                            upper[0].y - upper[1].y,
                            upper[0].z - upper[1].z);
      const nite::Point3f r(upper[5].x - upper[4].x,
                            upper[5].y - upper[4].y,
                            upper[5].z - upper[4].z);

      const float U = sqrt(u.x * u.x + u.y * u.y + u.z * u.z);
      const float L = sqrt(l.x * l.x + l.y * l.y + l.z * l.z);
      const float R = sqrt(r.x * r.x + r.y * r.y + r.z * r.z);

      left  = fabs((u.x * l.x + u.y * l.y + u.z * l.z) / (U * L)) < .5;
      right = fabs((u.x * r.x + u.y * r.y + u.z * r.z) / (U * R)) < .5;

      static bool left_old, right_old;
      static float d_old, x_old, z_old, ax_old, ly_old, az_old;

      if(left && right) {
        const float dx = upper[0].x - upper[5].x;
        const float dy = upper[0].y - upper[5].y;
        const float d  = sqrt(dx * dx + dy * dy);
        if(!left_old || !right_old) {
          d_old = d;
          ly_old = global::ly;
        }
        global::ly = ly_old * d_old / d;
      } else if(left) {
        if(!left_old)
          x_old = upper[0].x;
        if(upper[0].x < x_old - 100)
          global::dt_dump = -fabs(global::dt_dump);
        else if(upper[0].x > x_old + 100)
          global::dt_dump =  fabs(global::dt_dump);
        else {
          const double temp = global::dt_saved;
          global::dt_saved = global::dt_dump;
          global::dt_dump = temp;
        }
      } else if(right) {
        if(!right_old) {
          ax_old = global::ax;
          az_old = global::az;
          x_old = upper[5].x;
          z_old = upper[5].y;
        }
        global::ax = ax_old + upper[5].x - x_old;
        global::az = az_old + upper[5].y - z_old;
      }

      left_old  = left;
      right_old = right;
    } else
      head.z = 0;
  }
}

void track()
{
  glLoadIdentity();
  glTranslatef(0, 0, -5); // translate 5 meters

  if(head.z != 0) {
    glColor3f(1, 1, 0);
    glScalef(1./2000, 1./2000, 1./2000);

    glBegin(GL_LINE_STRIP);
    const float hx = head.x, hy = head.y, hz = head.z;
    for(int i = -16; i <= 48; ++i) {
      const float phi = M_PI * i / 32;
      glVertex3f(hx + 150 * cos(phi), hy + 150 * sin(phi), hz);
    }
    glVertex3f(neck.x, neck.y, neck.z);
    glEnd();

    glBegin(GL_LINE_STRIP);
    glVertex3f(upper[2].x, upper[2].y, upper[2].z);
    glVertex3f(   torso.x,    torso.y,    torso.z);
    glVertex3f(upper[3].x, upper[3].y, upper[3].z);
    glEnd();

    glBegin(GL_LINE_STRIP);
    glVertex3f(lower[2].x, lower[2].y, lower[2].z);
    glVertex3f(   torso.x,    torso.y,    torso.z);
    glVertex3f(lower[3].x, lower[3].y, lower[3].z);
    glEnd();

    glBegin(GL_LINE_STRIP);
    for(int i = 0; i < 6; ++i)
      glVertex3f(upper[i].x, upper[i].y, upper[i].z);
    glEnd();

    glBegin(GL_LINE_STRIP);
    for(int i = 0; i < 6; ++i)
      glVertex3f(lower[i].x, lower[i].y, lower[i].z);
    glEnd();

    if(left) {
      glLineWidth(7);
      glColor3f(.333, .333, 1);
      glBegin(GL_LINE_STRIP);
      glVertex3f(upper[0].x, upper[0].y, upper[0].z);
      glVertex3f(upper[1].x, upper[1].y, upper[1].z);
      glEnd();
      glColor3f(1, 1, 0);
      glLineWidth(1);
    }

    if(right) {
      glLineWidth(7);
      glColor3f(.333, .333, 1);
      glBegin(GL_LINE_STRIP);
      glVertex3f(upper[5].x, upper[5].y, upper[5].z);
      glVertex3f(upper[4].x, upper[4].y, upper[4].z);
      glEnd();
      glColor3f(1, 1, 0);
      glLineWidth(1);
    }
  }
}
#endif
