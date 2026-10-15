# Firmware (ESP32-S3)

Arduino sketches for the ESP32-S3 driving the CoreXY gantry through a CNC Shield V3 with TMC2209 drivers.

| Sketch | What it does |
|---|---|
| `circle_motors/` | Motion test: draws a circle repeatedly with CoreXY kinematics (Bresenham step sync). |
| `home_LS/` | Homes Y then X against the limit switches once at boot, with a back-off. |
| `move_XY_using_serial/` | Homes at boot, then accepts serial commands with a trapezoidal speed profile and soft limits of 0–49 cm. |

## Pins

| Signal | GPIO |
|---|---|
| X STEP / DIR | 1 / 2 |
| Y STEP / DIR | 41 / 42 |
| Driver EN (active LOW) | 21 |
| Limit X / Limit Y (NO, to GND, internal pull-up) | 15 / 16 |

## Serial commands (`move_XY_using_serial`, 115200 baud)

- `G <X> <Y>`: absolute move in cm
- `R <dX> <dY>`: relative move in cm
- `H`: re-home

Calibration: 400 steps/cm, max 120 mm/s, acceleration 600 mm/s².
