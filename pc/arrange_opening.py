"""The robot sets up the opening by itself, from all 30 checkers in the bear-off trays
(Nadav 14.11).

    py arrange_opening.py     (double-click)

  1. Put the checkers in the trays (place_trays.py shows you where).
  2. Enter = start    q = stop
  3. The robot drags them one by one (around the others): the columns farthest from the
     trays first, and in each column the checker at the edge first, then parks.
"""
import sys  # noqa: E402
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent / "lib"))  # the internal code lives in lib/

import rules
import ui
from place_opening import ALL_OUT


def order(L, board, frm):
    """The places of `board` in the order a person fills them (Nadav 14.11): the columns
    farthest from where the checkers come from first, and in each column from the edge
    of the board inward (the bottom checker first)."""
    src = [xy for xy, _ in L.place(frm).values()]
    sy = float(sum(y for _, y in src)) / max(1, len(src))
    items = sorted(L.place(board).items(), key=lambda kv: (-abs(kv[1][0][1] - sy), str(kv[0][0]), kv[0][1]))
    return [(where, i, xy, side) for (where, i), (xy, side) in items]


def outer_first(L, xy):
    """Which checker of a pile to lift first (smallest wins): in a bear-off tray the whole outer
    row (next to the board edge) before the inner one, and in a row the one nearest the lane
    (Nadav 14.11). The one exception: a checker standing right at the lane (the 15th of a tray
    spills into the inner row beside it) goes first, or it blocks the lane for all the others."""
    x, y = L.from_magnet(*xy)
    d = abs(x - L.xmax / 2)
    return (0 if d < 30 else 1, round(y / 10.0), d)


def arrange(b, L, frm, to, log=print, ex=None):
    """Drag every checker from the board `frm` (where they are, at their exact places) to
    the board `to`, in order(). Returns the executor (its warnings) and the number of drags.
    ex: an executor to use (the game's own, so its page shows the drags)."""
    from executor import Executor
    ex = ex or Executor(b, L, log=lambda *a: None)
    ex.set_from_board(frm)
    n = 0
    for where, i, xy, side in order(L, to, frm):
        src = [k for k in ex.places if str(k).endswith(side) and not isinstance(k, int) and ex.places[k]]
        if not src:
            log(f"   no {side} checker left to put on {where}")
            continue
        ex.move_checker(src[0], where, side, goal=tuple(xy), pick=lambda xy_, L=L: outer_first(L, xy_))
        n += 1
    ex.park(L.park())
    for w in ex.warnings:
        log("   warning: " + w)
    return ex, n


def main():
    from board import Board
    from layout import Layout
    L = Layout.load()
    print(__doc__.split("\n\n")[1])
    if input("All 30 checkers in the trays?\n   Enter = start\n   q = stop\n> ").strip().lower() == "q":
        return
    frm, to = ALL_OUT, rules.OPENING
    with Board() as b:
        b.home_if_needed()
        print("Arranging (30 drags, a few minutes)...")
        _, n = arrange(b, L, frm, to)
        print(f"   {n} drags done")
    print("Done. Look at the board: every checker should stand in its place.")


if __name__ == "__main__":
    ui.keep_open()
    main()
