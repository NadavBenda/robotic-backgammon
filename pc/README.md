# PC side

Talks to `firmware/board` over USB serial (no Wi-Fi). Needs `pip install pyserial opencv-python numpy`.

- `board.py`: `Board` class. Each call sends one command and waits for `DONE` (or raises `BoardError` on `ERR` / timeout), so commands never overlap. Units are mm.
- `console.py`: **the single place to control the board** (instead of the Serial Monitor). Type `H`, `G 100 200`, `M 1`, `C`, `?`; `C` saves and shows the photo. Normal mode shows results and warnings; type the toggle word (`DEV_TOGGLE` at the top, default `dev`) for developer mode with every board log line and timings.
- `gnubg.py`: GNU Backgammon as the move engine. `best_move(board, dice)` takes the 26-number board (`set board simple` format, from the side on roll) and returns `[Step(frm, to, hit)]` (BAR=25, OFF=0). `hit=True` = move the opponent checker to the bar first. Path to `gnubg-cli.exe` and strength (`PLIES`) at the top.
- `capture.py`: press Enter to grab a JPEG (XGA 1024x768) into `captures/` and show it.

```python
from board import Board
with Board() as b:          # finds the ESP32 port by itself, or Board("COM5")
    b.home()
    b.move(100, 200)
    b.magnet(True); b.move(150, 200); b.magnet(False)
    jpeg = b.capture()
```
