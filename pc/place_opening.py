"""Set up the opening position with no graphics on the glass: the robot shows you where.

    py place_opening.py     (double-click)

The magnet cart drives under each place of the opening position, one after the other.
You see the cart through the glass: put a checker right above it and press Enter.
The magnet then turns on for a moment, so the checker slides to the exact centre.
First the robot's 15 checkers (one colour), then your 15 (the other colour).
  Enter = placed    s = skip this one    b = back one    q = stop
At the end the cart parks in the middle lane (bar row), out of the camera's way.
"""
import sys  # noqa: E402
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent / "lib"))  # the internal code lives in lib/

import time

import rules
import ui
from board import Board
from helpers import magnet_on
from layout import Layout

PULSE_S = 0.4


def spots(L):
    """[(side, where, i, (x, y))] robot first, each point from the board edge inward."""
    out = []
    for side in ("robot", "human"):
        for (where, i), (xy, s) in sorted(L.place(rules.OPENING).items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
            if s == side:
                out.append((side, where, i, xy))
    return out


def name(side, where):
    if side == "robot":
        return f"robot point {where}"
    return f"your point {25 - where}"


def main():
    L = Layout.load()
    todo = spots(L)
    print(__doc__.split("\n\n")[1])
    with Board() as b:
        b.home_if_needed()
        k = 0
        while k < len(todo):
            side, where, i, (x, y) = todo[k]
            if k == 0 or todo[k - 1][0] != side:
                print("\n=== " + ("ROBOT's colour (15)" if side == "robot" else "YOUR colour (15)") + " ===")
            b.move(x, y)
            ans = input(f"[{k + 1}/30] {name(side, where)}, checker {i + 1}: put it above the cart, Enter ").strip().lower()
            if ans == "q":
                break
            if ans == "b":
                k = max(0, k - 1)
                continue
            if ans != "s":
                magnet_on(b)
                time.sleep(PULSE_S)
                b.magnet(False)
            k += 1
        b.move(*L.park())
    print("Done. Now: py game.py, and press 'Learn colours' the first time.")


if __name__ == "__main__":
    ui.keep_open()
    main()
