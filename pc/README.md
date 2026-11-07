# PC side

Talks to `firmware/board` over USB serial (no Wi-Fi). Needs `pip install pyserial opencv-python numpy`.

- `board.py`: `Board` class. Each call sends one command and waits for `DONE` (or raises `BoardError` on `ERR` / timeout), so commands never overlap. Units are mm.
- `capture.py`: press Enter to grab a JPEG (XGA 1024x768) into `captures/` and show it.

```python
from board import Board
with Board() as b:          # finds the ESP32 port by itself, or Board("COM5")
    b.home()
    b.move(100, 200)
    b.magnet(True); b.move(150, 200); b.magnet(False)
    jpeg = b.capture()
```
