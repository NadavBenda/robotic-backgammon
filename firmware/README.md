# Firmware (ESP32-S3)

Arduino sketches for the ESP32-S3 driving the CoreXY gantry through a CNC Shield V3 with TMC2209 drivers.

| Sketch | What it does |
|---|---|
| `circle_motors/` | Motion test: draws a circle repeatedly with CoreXY kinematics (Bresenham step sync). |
| `home_LS/` | Homes Y then X against the limit switches once at boot, with a back-off. |
| `move_XY_using_serial/` | Homes at boot, then accepts serial commands with a trapezoidal speed profile and soft limits of 0–49 cm. |

## Board settings check (`BoardCheck.h`)

Every sketch folder has a copy of `BoardCheck.h` (Arduino can't include from outside the sketch folder). Wrong Tools settings stop the compile with an `#error` that names the menu and the right value: board, PSRAM = OPI, Flash Mode = QIO, CPU 240 MHz, Partition = 16M Flash (3MB APP/9.9MB FATFS), USB Mode = Hardware CDC. `boardCheck()` at boot also checks flash size (16 MB) and PSRAM (8 MB) and prints a `[BOARD]` line.

## Pins

| Signal | GPIO |
|---|---|
| X STEP / DIR | 1 / 2 |
| Y STEP / DIR | 41 / 42 |
| Driver EN (active LOW) | 21 |
| Limit X / Limit Y (NO, to GND, internal pull-up) | 39 / 40 |
| Electromagnet (AOD4184 PWM) | 47 |

## Serial commands (`move_XY_using_serial`, 115200 baud)

- `G <X> <Y>`: absolute move in cm
- `R <dX> <dY>`: relative move in cm
- `H`: re-home
- `M 1` / `M 0`: electromagnet on / off

## Electromagnet safety (`Magnet.h`)

`Magnet` class: pin LOW at boot, a hardware `esp_timer` forces it off after 10 s on (works even while a move blocks `loop()`), and a leaky-bucket heat budget (60 s on, cools at half speed) refuses `on()` when hot. The magnet also turns off on a hard-limit hit.

Calibration: 400 steps/cm, max 120 mm/s, acceleration 600 mm/s².
