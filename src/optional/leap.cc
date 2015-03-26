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

class GRayLeapListener : public Leap::Listener {
  public:
    virtual void onConnect(const Leap::Controller&);
    virtual void onFrame  (const Leap::Controller&);
};

void GRayLeapListener::onConnect(const Leap::Controller& controller)
{
  controller.enableGesture(Leap::Gesture::TYPE_CIRCLE);
  controller.enableGesture(Leap::Gesture::TYPE_KEY_TAP);
}

void GRayLeapListener::onFrame(const Leap::Controller& controller)
{
  static int type = 0, direction = 1, direction_old = 0;
  static float d_old, x_old, z_old, ax_old, ly_old, az_old;

  const Leap::Frame    frame = controller.frame();
  const Leap::HandList hands = frame.hands();

  if(hands.count() == 2 &&
     hands[0].fingers().count() >= 1 &&
     hands[1].fingers().count() >= 1) {
    const Leap::FingerList lfingers = hands[0].fingers();
    const Leap::FingerList rfingers = hands[1].fingers();
    Leap::Vector lpos, rpos;

    for(int i = 0; i < lfingers.count(); ++i)
      lpos += lfingers[i].tipPosition();
    lpos /= (float)lfingers.count();

    for(int i = 0; i < rfingers.count(); ++i)
      rpos += rfingers[i].tipPosition();
    rpos /= (float)rfingers.count();

    const float d = lpos.distanceTo(rpos);
    if(d > 10.0) { // d may be NaN
      if(type != 2) {
        d_old  = d;
        ly_old = vis::ly;
        type   = 2;
      }
      vis::ly = ly_old * d_old / d;
    }
  } else if(hands.count() == 1 &&
            hands[0].fingers().count() >= 1) {
    const Leap::GestureList gestures = frame.gestures();
    bool detected = false;

    for(int i = 0; i < gestures.count(); ++i)
      // If a "Tap" gesture is detected, swap direction with
      // direction_old so we restore the previous direction
      if(gestures[i].type() == Leap::Gesture::TYPE_KEY_TAP) {
        detected = true;

        int tmp = direction_old;
        direction_old = direction;
        direction = tmp;
        break;
      }

    for(int i = 0; i < gestures.count(); ++i)
      // If a "Circle" gesture is detected, save the current state
      // only if we are in pause mode, then set direction to the
      // circle direction.
      if(gestures[i].type() == Leap::Gesture::TYPE_CIRCLE) {
	detected = true;

        if(direction == 0) direction_old = 0;
        Leap::CircleGesture circle = gestures[i];
        direction = circle.pointable().direction().angleTo(circle.normal())
                 <= Leap::PI/4 ? 1 : -1;
        break;
      }

    if(detected) {
      if(direction) {
        if(vis::direction == 0)
   	  vis::saved = 0;
        vis::direction = -direction;
      } else if(vis::direction) {
        vis::saved     = vis::direction;
        vis::direction = 0;
      }
      type = 0;
    } else {
      for(int i = 0; i < gestures.count(); ++i)
        if(gestures[i].type() == Leap::Gesture::TYPE_SWIPE) {
          const Leap::FingerList fingers = hands[0].fingers();
          Leap::Vector pos;

          for(int i = 0; i < fingers.count(); ++i)
            pos += fingers[i].tipPosition();
          pos /= (float)fingers.count();

          if(type != 1) {
            ax_old = vis::ax;
            az_old = vis::az;
            x_old  = -pos.x;
            z_old  =  pos.y;
            type   = 1;
          }
          vis::ax = ax_old + (-pos.x - x_old) / 2;
          vis::az = az_old + ( pos.y - z_old) / 2;

          break;
        }
    }
  } else
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
