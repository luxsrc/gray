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
#include <NiTE.h>

static bool first = true;

static nite::UserTracker tracker;
static nite::Point3f head;

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

    head.z = 0;
    for(int i = 0; i < users.getSize(); ++i) {
      const nite::UserData& user = users[i];

      if(user.isNew())
        tracker.startSkeletonTracking(user.getId());
      else if(nite::SKELETON_TRACKED == user.getSkeleton().getState()) {
        head = user.getSkeleton().getJoint(nite::JOINT_HEAD).getPosition();
        print("head: (%g, %g, %g)\n", head.x, head.y, head.z);
      }
    }
  }
}

void track()
{
  glLoadIdentity();
  glTranslatef(0, 0, -5); // translate 5 meters

  if(head.z != 0) {
    glPushMatrix();
    glTranslatef(head.x / 1000, head.y / 1000, head.z / 1000);
    glRotatef(-90, 1, 0, 0);
    glColor3f(1, 1, 0);
    glutSolidSphere(0.1, 32, 16);
    glPopMatrix();
  }
}
