"""The robot sets up the opening by itself, from all 30 checkers in the bear-off trays
(Nadav 14.11).

    py arrange_opening.py     (double-click)

  1. Put the checkers in the trays (place_trays.py shows you where).
  2. Enter = start    q = stop
  3. The robot drags them one by one (around the others), then parks and takes a photo.
     With detector.json (train_detector.py) it reads the photo and says what, if
     anything, is not where it should be: push that one by hand, or press Arrange on the
     game page.
"""
import sys  # noqa: E402
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent / "lib"))  # the internal code lives in lib/

import rules
import ui
from place_opening import ALL_OUT


def arrange(b, L, frm, to, log=print):
    """Drag every checker from the board `frm` (where they are, at their exact places) to
    the board `to`. Returns the executor (its warnings) and the number of drags."""
    from executor import Executor
    ex = Executor(b, L, log=lambda *a: None)
    ex.set_from_board(frm)
    n = ex.arrange(to, park=L.park())
    for w in ex.warnings:
        log("   warning: " + w)
    return ex, n


def check(b, c, L, want, log=print):
    """Photo + the trained detector: [text] where it differs from `want` ([] = all fine,
    None = no detector.json)."""
    import learned
    import vision
    from helpers import photo
    try:
        det = learned.Detector.load()
    except (OSError, ValueError, KeyError):
        return None
    img = photo(b)[0]
    got = learned.full(vision.summarize(det.find(img, c, L)[0], L))
    off = rules.off_count(want)
    exp = list(want) + [off[0], off[1]]
    out = []
    for i, (g, e) in enumerate(zip(got, exp)):
        if g != e:
            name = "bar" if i in (0, 25) else ("robot's tray" if i == 26 else "your tray") if i >= 26 else f"column {25 - i}"
            out.append(f"{name}: should be {abs(e)}, the camera sees {abs(g)}")
    return out


def main():
    import calib
    from board import Board
    from layout import Layout
    L = Layout.load()
    c = calib.load()
    print(__doc__.split("\n\n")[1])
    if input("All 30 checkers in the trays?  Enter = start   q = stop: ").strip().lower() == "q":
        return
    frm, to = ALL_OUT, rules.OPENING
    with Board() as b:
        b.home_if_needed()
        print("Arranging (30 drags, a few minutes)...")
        _, n = arrange(b, L, frm, to)
        print(f"   {n} drags done")
        bad = check(b, c, L, to)
    if bad is None:
        print("Done. (No detector.json, so no photo check: look at the board.)")
    elif not bad:
        print("Done. The camera sees every checker in its place.")
    else:
        print("Done, but the camera sees something else at:")
        for t in bad:
            print("   " + t)
        print("Push those by hand, or start game.py and press Arrange.")


if __name__ == "__main__":
    ui.keep_open()
    main()
