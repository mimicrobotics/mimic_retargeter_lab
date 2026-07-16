"""Smoke test: read from a SpaceMouse and print 6DOF pose data.

Prerequisites:
  1. SpaceMouse connected via USB.
  2. On Linux, HID permissions are needed. Add a udev rule:
       echo 'SUBSYSTEM=="hidraw", ATTRS{idVendor}=="256f", MODE="0666"' \
         | sudo tee /etc/udev/rules.d/99-spacemouse.rules
       sudo udevadm control --reload-rules && sudo udevadm trigger

Usage:
  python scripts/test_spacemouse.py
"""

import pyspacemouse

print("Opening SpaceMouse (auto-detect)...")
device = pyspacemouse.open(nonblocking=True)
print("Connected. Move the knob to see data. Press Ctrl+C to exit.\n")

try:
    while True:
        state = device.read()

        lin = f"x={state.x:+.3f}  y={state.y:+.3f}  z={state.z:+.3f}"
        ang = f"roll={state.roll:+.3f}  pitch={state.pitch:+.3f}  yaw={state.yaw:+.3f}"
        btn = f"buttons={list(state.buttons)}" if state.buttons else "buttons=[]"

        print(f"  linear: {lin}  |  angular: {ang}  |  {btn}", end="\r")

except KeyboardInterrupt:
    print("\n\nDone.")
finally:
    device.__exit__(None, None, None)
