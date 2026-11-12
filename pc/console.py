"""Console for the board: the single place to control it (instead of the Serial Monitor).

    py console.py            # finds the ESP32 port by itself
    py console.py COM5

Type board commands as usual: H, G 100 200, R 10 0, M 1, M 0, C, ?
  C saves the photo in captures/ and shows it; after calibration also the photo
  straightened to a top view (1 px = 1 mm, saved as *_top.jpg).
  q = quit.
Two modes:
  normal    only results (OK / ERR), warnings (!) and answers such as POS
  developer also every log line from the board (#) and how long each command took
Switch with the secret word below.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from board import Board, BoardError

DEV_TOGGLE = "dev"  # the word that switches normal <-> developer mode

TIMEOUTS = {"H": 60, "G": 30, "R": 30, "C": 10}
out = Path(__file__).parent / "captures"
dev = False


def log(line):
    if dev and line not in ("DONE", ""):
        print("   <", line)


def topdown_or_none(img):
    try:
        import calib
        return calib.topdown(img, calib.load())
    except (OSError, ValueError, KeyError):
        return None   # no calib.json yet


def run(b, line):
    t0 = time.time()
    if dev:
        print("   >", line)
    try:
        info = b.command(line, timeout=TIMEOUTS.get(line[:1].upper(), 5))
    except BoardError as e:
        print("ERR", str(e).split("ERR", 1)[-1].strip())
        return
    for x in info:
        if isinstance(x, bytes):
            out.mkdir(exist_ok=True)
            path = out / time.strftime("%Y%m%d_%H%M%S.jpg")
            path.write_bytes(x)
            img = cv2.imdecode(np.frombuffer(x, np.uint8), cv2.IMREAD_COLOR)
            print(f"photo {img.shape[1]}x{img.shape[0]}, {len(x) // 1024} KB -> {path.name}")
            cv2.imshow("board", img)
            flat = topdown_or_none(img)
            if flat is not None:   # after calibration: also the straightened top view
                cv2.imwrite(str(path.with_name(path.stem + "_top.jpg")), flat)
                cv2.imshow("board from above (1 px = 1 mm)", flat)
            cv2.waitKey(1)
        elif not dev:
            print(x)
    print(f"OK ({time.time() - t0:.2f} s)" if dev else "OK")


def main():
    global dev
    b = Board(sys.argv[1] if len(sys.argv) > 1 else None, on_event=lambda e: print("!!", e[1:].strip()), on_log=log)
    print(f"Connected on {b.ser.port}. Commands: H, G x y, R dx dy, M 1/0, C, ?   q = quit")
    with b:
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if line.lower() == "q":
                break
            if line == DEV_TOGGLE:
                dev = not dev
                print("developer mode" if dev else "normal mode")
                continue
            run(b, line)
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


if __name__ == "__main__":
    main()
