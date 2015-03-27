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

#include "../gray.h"

#include <Leap.h>

static int type = 0, direction = 1, direction_old = 0;
static float x_old, d_old, y_old, ax_old, ly_old, az_old;

static void zoom(const Leap::Vector &l, const Leap::Vector &r)
{
  float d = l.distanceTo(r);
  if(isnan(d))
    return;

  if(type != 1) {
    type = 1;
    d_old = d;
    ly_old = vis::ly;
  }
  vis::ly = ly_old * d_old / d;
}

static void rotate(const Leap::Vector &p)
{
  if(type != 2) {
    type = 2;

    x_old = p.x;
    y_old = p.y;

    ax_old = vis::ax;
    az_old = vis::az;
  }
  vis::ax = ax_old - (p.x - x_old) / 2;
  vis::az = az_old + (p.y - y_old) / 2;
}

class GRayLeapListener : public Leap::Listener {
  public:
    virtual void onConnect(const Leap::Controller&);
    virtual void onFrame  (const Leap::Controller&);
};

void GRayLeapListener::onConnect(const Leap::Controller& controller)
{
  controller.enableGesture(Leap::Gesture::TYPE_CIRCLE);
}

void GRayLeapListener::onFrame(const Leap::Controller& controller)
{
  const Leap::Frame    frame = controller.frame();
  const Leap::HandList hands = frame.hands();

  if(hands.count() == 2) {
    int i = hands[1].isLeft();
    Leap::Hand l = hands[i];
    Leap::Hand r = hands[1-i];
    if(l.grabStrength() > 0.5 &&
       r.grabStrength() > 0.5) {
      zoom(l.palmPosition(), r.palmPosition());
      return;
    }
  } else if(hands.count() == 1) {
     Leap::Hand h = hands[0];
     Leap::FingerList f = h.fingers().extended();
     if(f.count() == 1) {
       type = 3;
       Leap::GestureList g = frame.gestures();
       for(int i = 0; i < g.count(); ++i)
	 if(g[i].type() == Leap::Gesture::TYPE_CIRCLE) {
	   if(direction == 0) direction_old = 0;
	   Leap::CircleGesture c = g[i];
	   if(c.pointable().direction().angleTo(c.normal()) <= Leap::PI/2)
	     vis::direction = 1;
	   else
	     vis::direction = -1;
	   return;
	 }
       if(vis::direction) {
	 vis::saved = vis::direction;
	 vis::direction = 0;
       }
       return;
     }
     if(h.grabStrength() > 0.5) {
       rotate(h.palmPosition());
       return;
     }
  }

  type = 0;
}

static Leap::Controller *controller = NULL;
static GRayLeapListener *listener   = NULL;

static void setup()
{
  print("Leaping motion...");
  controller = new Leap::Controller;
  listener   = new GRayLeapListener;
  controller->addListener(*listener);
  print(" DONE\n");
}

static void cleanup()
{
  print("Cleanup...");
  controller->removeListener(*listener);
  print(" DONE\n");
}

void vis::sense()
{
  if(!controller && !atexit(cleanup)) setup();
}
