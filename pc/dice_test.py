"""Try the camera on real dice, before using them in a game (Nadav 15.11).

    py dice_test.py     (double-click)

  1. Take the dice off the glass. Enter = a photo of the board without dice.
  2. Roll the two dice anywhere on the glass (not on a checker). Enter = the camera reads them.
  3. Say what the dice really show (the faces on top), so the score is kept.
Every try is saved in dice_test/ (the two photos and the top view with what was found:
green box = a die, red rings = its pips), with a line in dice_test/results.csv. Nothing moves
except the magnet cart, which parks out of the camera's way at the start.
"""
import sys  # noqa: E402
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent / "lib"))  # the internal code lives in lib/

import time
from pathlib import Path

import cv2

import dice
import ui

OUT = Path(__file__).parent / "dice_test"


def save(path, img):
    cv2.imencode(".jpg", img)[1].tofile(str(path))   # works with Hebrew paths


def main():
    import calib
    from board import Board
    from helpers import photo
    from layout import Layout
    L = Layout.load()
    c = calib.load()
    OUT.mkdir(exist_ok=True)
    right = total = 0
    with Board() as b:
        b.home_if_needed()
        b.move(*L.park())
        before = None
        while True:
            if before is None:
                if input("\nTake the dice off the glass.\n   Enter = photo of the board without dice\n   q = stop\n> ").strip().lower() == "q":
                    break
                before = photo(b)[0]
            a = input("\nRoll the two dice on the glass.\n   Enter = read them\n   n = the board changed: a new photo without dice first\n   q = stop\n> ").strip().lower()
            if a == "q":
                break
            if a == "n":
                before = None
                continue
            after = photo(b)[0]
            dbg = []
            t0 = time.time()
            got = dice.read(before, after, c, L, dbg)
            print(f"The camera reads: {f'{got[0]}-{got[1]}' if got else 'nothing clear'}   ({time.time() - t0:.1f} s)")
            real = input("What do the dice really show?\n   Enter = the reading is right\n   two numbers = what they show, like 5 2\n   s = skip this one\n> ").strip().lower()
            if real == "s":
                continue
            truth = dice.parse(real) if real else got
            if truth is None:
                print("   (no reading and no numbers typed: skipped)")
                continue
            ok = got is not None and sorted(got) == sorted(truth)
            total += 1
            right += ok
            tag = time.strftime("%Y%m%d-%H%M%S")
            save(OUT / f"{tag}_before.jpg", before)
            save(OUT / f"{tag}_after.jpg", after)
            if dbg and dbg[0] is not None:
                save(OUT / f"{tag}_found.jpg", dbg[0])
            with open(OUT / "results.csv", "a", encoding="utf-8") as f:
                f.write(f"{tag},{''.join(map(str, got)) if got else ''},{''.join(map(str, truth))},{int(ok)}\n")
            print(f"   {'right' if ok else 'WRONG'}.  So far {right} of {total} right.")
    print(f"\nDone: {right} of {total} right. The photos are in dice_test/")


if __name__ == "__main__":
    ui.keep_open()
    main()
