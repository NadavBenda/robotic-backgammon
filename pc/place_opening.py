"""Set up the opening position with no graphics on the glass: the robot shows you where.

    py place_opening.py     (double-click)

The magnet cart drives under each place of the opening position, one after the other,
around the checkers already placed.
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

import paths
import rules
import ui
from board import Board
from helpers import magnet_on
from layout import Layout

PULSE_S = 0.4


ALL_OUT = [0] * 26     # every checker borne off: 15 + 15 in the bear-off trays


def spots(L, board=None):
    """[(side, where, i, (x, y))] robot first, each point (or tray) from the board edge inward."""
    out = []
    for side in ("robot", "human"):
        for (where, i), (xy, s) in sorted(L.place(rules.OPENING if board is None else board).items(),
                                          key=lambda kv: (str(kv[0][0]), kv[0][1])):
            if s == side:
                out.append((side, where, i, xy))
    return out


def name(side, where):
    if str(where).startswith("off_"):
        return ("robot's" if side == "robot" else "your") + " bear-off tray"
    if side == "robot":
        return f"robot point {where}"
    return f"your point {25 - where}"


def main(board=None, doc=None, done="Done. Now: py game.py, and press 'Learn colours' the first time."):
    L = Layout.load()
    todo = spots(L, board)
    print((doc or __doc__).split("\n\n")[1])
    with Board() as b:
        b.home_if_needed()
        k = 0
        cart, placed = None, {}   # placed: k -> (x, y) of the checkers already on the glass
        while k < len(todo):
            side, where, i, (x, y) = todo[k]
            if k == 0 or todo[k - 1][0] != side:
                print("\n=== " + ("ROBOT's colour (15)" if side == "robot" else "YOUR colour (15)") + " ===")
            # magnet off: around the checkers already placed (they could be dragged along)
            paths.travel(b, cart, (x, y), list(placed.values()), L.xmax, L.ymax)
            cart = (x, y)
            ans = input(f"[{k + 1}/30] {name(side, where)}, checker {i + 1}: put it above the cart, Enter ").strip().lower()
            if ans == "q":
                break
            if ans == "b":
                k = max(0, k - 1)
                placed.pop(k, None)
                continue
            if ans != "s":
                magnet_on(b)
                time.sleep(PULSE_S)
                b.magnet(False)
                placed[k] = (x, y)
            k += 1
        paths.travel(b, cart, L.park(), list(placed.values()), L.xmax, L.ymax)
    print(done)


if __name__ == "__main__":
    ui.keep_open()
    main()
