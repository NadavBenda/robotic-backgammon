"""Move checkers by hand-picked start and goal, along a path that avoids the others.

    py move.py

Every photo is shown corrected (seen from straight above, from calib.json).

1. A photo of the whole board. Click every checker on it (click again to remove one),
   Enter when done. (No automatic detection for now.)
2. Click the checker to move, then where it should go. The planned path is drawn in blue.
   Enter = go, r = pick again, q = finish.
3. After the move a photo from beside the goal: click where the checker really is.
   Press d before that if a neighbour moved too (then you fix the marks on a new photo).
Repeat from 2; the list of checkers is kept up to date. At the end: how many moves landed
within 5 mm and how many disturbed a neighbour. Photos: captures/move/.
"""
import sys  # noqa: E402
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent / "lib"))  # the internal code lives in lib/

import time
from pathlib import Path

import cv2
import numpy as np

import calib
import ui
from board import Board
from helpers import photo, step_aside
from paths import CHECKER_MM, NoPath, XMAX, YMAX, drag, length, plan

WIN = "move"
PARK = (205.0, 420.0)   # same spot as the game (Layout.park): the camera sees the whole board
AWAY = 80
OK_MM = 5.0
SHOTS = Path(__file__).parent / "captures" / "move"


class _Aside:  # what calibrate.step_aside expects
    away, xmax, ymax = AWAY, XMAX, YMAX


def window(c, img, lines, marks, toggle=False, path=None, extra_keys=""):
    """Show the photo with the checkers (green) and an optional path (blue).
    toggle=True: clicks add/remove checkers in marks, Enter returns 'enter'.
    toggle=False: returns ('click', (x, y)) on a click.
    Keys in extra_keys and q are returned as they are."""
    dy = ui.bar_h(lines)
    out = []

    def on_mouse(ev, u, v, *_):
        if ev != cv2.EVENT_LBUTTONDOWN or v < dy:
            return
        x, y = (float(t) for t in c.px_to_mm(u, v - dy))
        if toggle:
            near = [m for m in marks if np.hypot(m[0] - x, m[1] - y) < CHECKER_MM / 2]
            if near:
                marks.remove(near[0])
            else:
                marks.append((x, y))
        else:
            out.append((x, y))

    ui.show(WIN, ui.with_bar(img, lines))
    cv2.setMouseCallback(WIN, on_mouse)
    while True:
        view = ui.with_bar(img, lines)
        for x, y in marks:
            u, v = c.mm_to_px(x, y)
            ui.draw_bottom(view, c.px_to_mm, c.mm_to_px, float(u), float(v), dy=dy)
        if path:
            px = np.int32([[float(t) for t in c.mm_to_px(*p)] for p in path])
            px[:, 1] += dy
            cv2.polylines(view, [px], False, (255, 0, 0), 3, cv2.LINE_AA)
        ui.show(WIN, view)
        k = cv2.waitKey(30) & 0xFF
        if out:
            return "click", out[0]
        if k in (13, 10):
            return "enter", None
        if k == ord("q"):
            return "quit", None
        if k < 128 and chr(k) in extra_keys:
            return chr(k), None


def snap(p, marks):
    """The marked checker nearest to the click, if the click was on one."""
    near = [m for m in marks if np.hypot(m[0] - p[0], m[1] - p[1]) < CHECKER_MM / 2]
    return min(near, key=lambda m: np.hypot(m[0] - p[0], m[1] - p[1])) if near else None


def shot(b, at, c):
    """Photo, saved raw; returned corrected to the top view."""
    b.move(*at)
    time.sleep(0.3)
    img, jpeg = photo(b)
    shot.n = getattr(shot, "n", 0) + 1
    (SHOTS / f"{shot.n:03d}.jpg").write_bytes(jpeg)
    return calib.view(img, c, xmax=XMAX, ymax=YMAX)[0]


def main():
    cal = calib.load()
    c = calib.Flat(cal)   # pixel <-> mm of the corrected pictures
    SHOTS.mkdir(parents=True, exist_ok=True)
    marks, results = [], []
    with Board() as b:
        b.home_if_needed()
        img = shot(b, PARK, cal)
        ans, _ = window(c, img, ["Click every checker on the board (again = remove)",
                                 "Enter = done   q = quit"], marks, toggle=True)
        if ans == "quit":
            return
        while True:
            img = shot(b, PARK, cal)
            ans, p = window(c, img, [f"Move {len(results) + 1}: click the checker to move",
                                     "q = finish"], marks)
            if ans == "quit":
                break
            start = snap(p, marks)
            if start is None:
                print("  that is not a marked checker; click on a green one")
                continue
            ans, goal = window(c, img, ["Now click where it should go", "q = finish"], marks)
            if ans == "quit":
                break
            try:
                path = plan(start, goal, [m for m in marks if m != start])
            except NoPath as e:
                print("  no path:", e)
                continue
            ans, _ = window(c, img, [f"Path: {len(path) - 1} line(s), {length(path):.0f} mm",
                                     "Enter = go   r = pick again   q = finish"],
                            marks, path=path, extra_keys="r")
            if ans == "quit":
                break
            if ans == "r":
                continue
            drag(b, path)
            img = shot(b, step_aside(*goal, _Aside), cal)
            flag = {"d": False}
            while True:
                ans, seen = window(c, img, ["Click where the checker is now (d = a neighbour moved)",
                                            "d marked" if flag["d"] else ""],
                                   [m for m in marks if m != start], extra_keys="d")
                if ans == "d":
                    flag["d"] = True
                elif ans in ("click", "quit"):
                    break
            if ans != "click":
                break
            err = float(np.hypot(seen[0] - goal[0], seen[1] - goal[1]))
            results.append((err, flag["d"]))
            marks[marks.index(start)] = seen
            print(f"  {len(results):2d}: ({start[0]:.0f},{start[1]:.0f}) -> ({goal[0]:.0f},{goal[1]:.0f}) "
                  f"via {len(path) - 1} line(s): landed {err:.1f} mm off "
                  f"(dx {seen[0] - goal[0]:+.1f}, dy {seen[1] - goal[1]:+.1f}; "
                  f"last line heading {np.degrees(np.arctan2(path[-1][1] - path[-2][1], path[-1][0] - path[-2][0])):.0f} deg)"
                  + ("  NEIGHBOUR MOVED" if flag["d"] else ""))
            if flag["d"]:
                print("  re-mark the checkers that moved:")
                img = shot(b, PARK, cal)
                window(c, img, ["Fix the checkers that moved (click = add/remove)",
                                "Enter = done"], marks, toggle=True)
        b.magnet(False)
    cv2.destroyAllWindows()
    if results:
        good = sum(e <= OK_MM for e, _ in results)
        bumps = sum(d for _, d in results)
        print(f"\n{len(results)} moves: {good} within {OK_MM:.0f} mm, {bumps} disturbed a neighbour, "
              f"worst {max(e for e, _ in results):.1f} mm")


if __name__ == "__main__":
    ui.keep_open()
    main()
