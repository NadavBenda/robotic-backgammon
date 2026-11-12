"""Check that the board map (layout.py) sits on the real board, and fix it in 4 clicks.

    py teach_layout.py

1. The magnet parks, a photo is taken, and every slot of the layout is drawn on it
   (green = points, blue = bar, yellow = bear-off trays). If they sit on the drawn board,
   press q: nothing to do.
2. Otherwise click, in this order, the place where the first checker (at the board edge) of
   point 1, point 12, point 13 and point 24 really is (the robot's numbering: 1 = its
   deepest home point, bear-off side). Enter after each click.
   A best-fit (shift + rotation + scale + mirror) is computed and saved in layout.json
   as "affine"; the slots are drawn again so you can check. s = skip a point.
Needs calib.json (py calibrate.py). The photo is shown corrected (top view).
"""
import cv2

import calib
import ui
from board import Board
from calibrate import ask, photo
from layout import Layout, fit_affine

WIN = "layout"
REFS = [(1, "point 1 (robot's deepest home point)"), (12, "point 12"),
        (13, "point 13"), (24, "point 24 (human's deepest home point)")]


def overlay(img, c, L):
    view = img.copy()
    for where, i, (x, y) in L.all_slots(depth=6):
        col = (0, 200, 0) if isinstance(where, int) else (255, 120, 0) if "bar" in where else (0, 210, 230)
        u, v = c.mm_to_px(x, y)
        cv2.circle(view, (int(u), int(v)), 6, col, 2, cv2.LINE_AA)
        if i == 0 and isinstance(where, int):
            cv2.putText(view, str(where), (int(u) + 8, int(v) - 8), ui.FONT, 0.5, col, 1, cv2.LINE_AA)
    return view


def main():
    cal = calib.load()
    L = Layout.load()
    with Board() as b:
        b.home_if_needed()
        b.move(*L.park())
        img, _ = photo(b)
    img, c = calib.view(img, cal, xmax=L.xmax, ymax=L.ymax)   # corrected top view
    pick = ask(overlay(img, c, L), None, "Does the map sit on the board? q = yes, done.  click = no, fix it",
               (c.px_to_mm, c.mm_to_px), "click anywhere = fix   q = done")
    if pick == "quit":
        return
    nominal, measured = [], []
    for p, name in REFS:
        ans = ask(img, None, f"click the first checker place (board edge) of {name}",
                  (c.px_to_mm, c.mm_to_px), "Enter=OK  s=skip  q=stop")
        if ans == "quit":
            return
        if ans == "skip":
            continue
        measured.append(tuple(float(t) for t in c.px_to_mm(*ans)))
        nominal.append(L._nominal(p, 0))
    if len(nominal) < 3:
        print("Need at least 3 points.")
        return
    L.affine, rms = fit_affine(nominal, measured)
    print(f"Fit error {rms:.1f} mm RMS")
    ans = ask(overlay(img, c, L), None, f"New map (error {rms:.1f} mm). Enter = save, q = do not save",
              (c.px_to_mm, c.mm_to_px), "Enter=save  q=cancel")
    if ans != "quit":
        L.save()
        print("Saved to layout.json")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ui.keep_open()
    main()
