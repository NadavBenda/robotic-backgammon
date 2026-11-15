"""Mark the columns on the glass: the robot carries one checker to the first place of every
column, and you draw your mark on the glass beside it.

    py mark_columns.py     (double-click)

Put ONE checker on the glass above the cart when asked. The robot then drags it to the edge
place of each column in turn (left side far to near, then right side near to far) and stops.
The checker is exactly where that column's first checker stands, so its two sides (toward
the neighbouring columns) are where the column starts and ends. Draw your mark in the strip
outside the board, beside the checker, then press Enter for the next column.
Two more stops, one per side, put the checker's CENTRE on the line between the bear-off
trays and the board: draw that line level with the centre of the checker.
  Enter = next    b = back one    q = stop
At the end the cart goes to the park place; take the checker off the glass.
"""
import sys  # noqa: E402
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent / "lib"))  # the internal code lives in lib/

import time

import executor
import paths
import ui
from board import Board
from helpers import magnet_on
from layout import Layout

PULSE_S = 0.4


def stops(L):
    """[(label, (x, y))] in the order the checker visits them: down the left side, across,
    up the right side. Each column at its edge place (checker 1 of that column); the bear-off
    line half way between the inner tray row and the first board row."""
    tray_y = max(L.off_rows_y)                         # the tray row next to the board
    first_y = L._row_y(0)                              # the board row next to the trays
    line_y = (tray_y + first_y) / 2
    left = [p for p in range(1, 13)]                   # points 1..12 are on the left
    right = [p for p in range(13, 25)]
    by_y = lambda p: L._plain(p, 0)[1]
    left.sort(key=by_y, reverse=True)                  # far side first (picture y large = far)
    right.sort(key=by_y)                               # then back up from the near side
    out = [(f"column {25 - p} (robot's {p}), left side", L.slot(p, 0)) for p in left]
    out.append(("bear-off line, LEFT side: draw it level with the checker's CENTRE",
                L.to_magnet(L.edge_x, line_y)))
    out.append(("bear-off line, RIGHT side: draw it level with the checker's CENTRE",
                L.to_magnet(L.xmax - L.edge_x, line_y)))
    out += [(f"column {25 - p} (robot's {p}), right side", L.slot(p, 0)) for p in right]
    return out


def main():
    L = Layout.load()
    todo = stops(L)
    print(__doc__.split("\n\n")[1])
    print(__doc__.split("\n\n")[2])
    with Board() as b:
        b.home_if_needed()
        b.drag_profile(*executor.DRAG)
        cart = todo[0][1]
        b.move(*cart)
        input("\nPut ONE checker on the glass right above the cart, then Enter ")
        magnet_on(b)
        time.sleep(PULSE_S)
        b.magnet(False)
        k = 0
        while k < len(todo):
            label, xy = todo[k]
            if xy != cart:
                paths.drag(b, [cart, xy], xmax=L.xmax, ymax=L.ymax)
                cart = xy
            ans = input(f"[{k + 1}/{len(todo)}] {label}: mark it, Enter ").strip().lower()
            if ans == "q":
                break
            k = max(0, k - 1) if ans == "b" else k + 1
        paths.travel(b, cart, L.park(), [], L.xmax, L.ymax)   # the only checker is where the cart starts
    print("Done. Take the checker off the glass.")


if __name__ == "__main__":
    ui.keep_open()
    main()
