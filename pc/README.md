# PC side

Talks to `firmware/board` over USB serial (no Wi-Fi). Needs `pip install pyserial opencv-python numpy`.

- `board.py`: `Board` class. Each call sends one command and waits for `DONE` (or raises `BoardError` on `ERR` / timeout), so commands never overlap. Units are mm.
- `console.py`: **the single place to control the board** (instead of the Serial Monitor). Type `H`, `G 100 200`, `M 1`, `C`, `?`; `C` saves and shows the photo. Normal mode shows results and warnings; type the toggle word (`DEV_TOGGLE` at the top, default `dev`) for developer mode with every board log line and timings.
- `gnubg.py`: GNU Backgammon as the move engine. `best_move(board, dice)` takes the 26-number board (`set board simple` format, from the side on roll) and returns `[Step(frm, to, hit)]` (BAR=25, OFF=0). `hit=True` = move the opponent checker to the bar first. Path to `gnubg-cli.exe` and strength (`PLIES`) at the top.
- `calibrate.py` + `calib.py`: camera calibration by the robot itself. One checker rides on the magnet over a 6x6 grid; at each point a photo is taken and the checker found (Hough circle, you confirm with Enter or click). The green ellipse is where the round bottom of the checker should appear there: click so it sits on the bottom. Fits pixel <-> mm (lens distortion + perspective + mirror) into `calib.json` and prints the error. `py calibrate.py test`: click a checker in the photo and the magnet goes there.
- `swap.py`: finds two checkers in a photo and swaps them through a free spot beside them (A -> spot, B -> A, spot -> B), for a demo. `--times 6` swaps back and forth, `--auto` skips the window when exactly two are found. Stops `--overshoot 3` mm past the target before letting go; `--wiggle 3` makes a small circle after grabbing; `--drag 120,300` sets the drag speed/accel.
- `drag_test.py`: click a checker and a target; drags it once per profile, alternating A->B / B->A, at 120/160/200/250 mm/s and 300/800 mm/s^2 (skips higher accelerations once one fails), and after each drag shows a photo where you click the checker (no automatic detection), then lets you test the next checker. Prints a table per checker (+ `captures/drag_test.csv`) and leaves the fastest profile that held for all of them set on the board.
- All scripts home only when the board says `HOMED 0` (after a reset).
- `paths.py`: `plan(start, goal, others)` returns the corners of a drag path whose checker keeps 30 mm (centre to centre) from every other checker (A* on a 5 mm grid, then corners cut, so usually 1-3 straight lines); `drag(b, path)` drives it with the magnet on and a 3 mm overshoot. Raises `NoPath` when the checkers block every way.
- `move.py`: click all checkers once, then pick a checker and a goal; shows the planned path, drives it, and you click where it landed. Counts moves within 5 mm and moves that disturbed a neighbour (the 10/10 test).
- `ui.py`: text strip above the photo, the expected checker-bottom ellipse, and `keep_open()` so a double-clicked script waits for Enter at the end (also after an error).
- `capture.py`: press Enter to grab a JPEG (XGA 1024x768) into `captures/` and show it.

```python
from board import Board
with Board() as b:          # finds the ESP32 port by itself, or Board("COM5")
    b.home()
    b.move(100, 200)
    b.magnet(True); b.move(150, 200); b.magnet(False)
    jpeg = b.capture()
```
