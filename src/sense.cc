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
#include <cmath>
#include <NiTE.h>

static bool first = true;

static nite::UserTracker tracker;
static nite::Point3f head, neck, torso, upper[6], lower[6];

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

    glBegin(GL_LINE_STRIP);
    const float hx = head.x / 1000, hy = head.y / 1000, hz = head.z / 1000;
    for(int i = -16; i <= 48; ++i) {
      const float phi = M_PI * i / 32;
      glVertex3f(hx + 0.1 * cos(phi), hy + 0.1 * sin(phi), hz);
    }
    glVertex3f(neck.x / 1000, neck.y / 1000, neck.z / 1000);
    glEnd();

    glBegin(GL_LINE_STRIP);
    glVertex3f(upper[2].x / 1000, upper[2].y / 1000, upper[2].z / 1000);
    glVertex3f(   torso.x / 1000,    torso.y / 1000,    torso.z / 1000);
    glVertex3f(upper[3].x / 1000, upper[3].y / 1000, upper[3].z / 1000);
    glEnd();

    glBegin(GL_LINE_STRIP);
    glVertex3f(lower[2].x / 1000, lower[2].y / 1000, lower[2].z / 1000);
    glVertex3f(   torso.x / 1000,    torso.y / 1000,    torso.z / 1000);
    glVertex3f(lower[3].x / 1000, lower[3].y / 1000, lower[3].z / 1000);
    glEnd();

    glBegin(GL_LINE_STRIP);
    for(int i = 0; i < 6; ++i)
      glVertex3f(upper[i].x / 1000, upper[i].y / 1000, upper[i].z / 1000);
    glEnd();

    glBegin(GL_LINE_STRIP);
    for(int i = 0; i < 6; ++i)
      glVertex3f(lower[i].x / 1000, lower[i].y / 1000, lower[i].z / 1000);
    glEnd();
  }
}
