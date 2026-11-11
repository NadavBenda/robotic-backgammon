"""Find two checkers on the board and swap them (demo / video).

    py swap.py                # one swap, confirm in the window
    py swap.py --times 6      # swap back and forth 6 times
    py swap.py --times 6 --auto   # no window when exactly 2 checkers are found

Needs calib.json (py calibrate.py). Before each photo the magnet waits at --park.
The window shows the checkers found: green ellipse = where the round bottom of a checker
there should appear (from calib.json), red = the two that will swap. Enter = swap the two shown
(or the two you clicked), click = pick a checker (click near one found, or anywhere
to mark it yourself), r = new photo, q = stop.

The swap goes through a free spot beside the two, so the checkers never touch:
A -> spot, B -> A, spot -> B. Each new swap starts from a fresh photo, so small
placement errors do not add up. Before letting go the magnet goes --overshoot mm (3)
past the target, because the checker lags a little behind it.
"""
import argparse
import time

import cv2
import numpy as np

import calib
import ui
from board import Board
from calibrate import find_circles, magnet_on, photo

WIN = "swap"
CHECKER_MM = 26.0


def detect(c, img, xmax, ymax):
    """Checkers in the photo as [(u, v, x_mm, y_mm)], inside the board, no doubles."""
    found = []
    for u, v, _ in find_circles(img):
        x, y = (float(t) for t in c.px_to_mm(u, v))
        if not (-10 <= x <= xmax + 10 and -10 <= y <= ymax + 10):
            continue
        if any(np.hypot(x - f[2], y - f[3]) < CHECKER_MM * 0.6 for f in found):
            continue
        found.append((u, v, x, y))
    return found


def choose(c, img, found):
    """Window: returns two (x_mm, y_mm), 'again' or 'quit'."""
    picked = []
    maps = (c.px_to_mm, c.mm_to_px)
    lines = ["", "click 2 checkers  Enter=swap red  r=new photo  q=stop"]
    dy = ui.bar_h(lines)

    def on_mouse(ev, u, v, *_):
        if ev != cv2.EVENT_LBUTTONDOWN or v < dy:
            return
        v -= dy
        near = [f for f in found if np.hypot(f[0] - u, f[1] - v) < 25]
        f = min(near, key=lambda f: np.hypot(f[0] - u, f[1] - v)) if near else \
            (u, v, *(float(t) for t in c.px_to_mm(u, v)))
        picked.append(f)
        del picked[:-2]  # keep the last two clicks

    ui.show(WIN, ui.with_bar(img, lines))
    cv2.setMouseCallback(WIN, on_mouse)
    while True:
        pair = picked if len(picked) == 2 else (found if len(found) == 2 else [])
        lines[0] = f"{len(found)} checkers found" + ("" if pair else ", pick 2")
        view = ui.with_bar(img, lines)
        for u, v, x, y in found:
            ui.draw_bottom(view, *maps, u, v, dy=dy)
            cv2.putText(view, f"{x:.0f},{y:.0f}", (int(u) + 22, int(v) + dy),
                        ui.FONT, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        for u, v, *_ in pair:
            ui.draw_bottom(view, *maps, u, v, color=(0, 0, 255), dy=dy, thick=3)
        ui.show(WIN, view)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 10) and pair:
            return [(f[2], f[3]) for f in pair]
        if k == ord("r"):
            return "again"
        if k == ord("q"):
            return "quit"


def _dist_to_segment(p, a, b):
    p, a, b = map(np.asarray, (p, a, b))
    t = np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-9), 0, 1)
    return float(np.hypot(*(p - (a + t * (b - a)))))


def free_spot(a, b, args):
    """A spot where A can wait so that no move passes over a standing checker:
    A -> spot must miss B, B -> A must miss the spot, spot -> B must miss A (B now)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    clear = CHECKER_MM * 1.3  # centre-to-path distance that keeps two checkers apart
    near = min(clear, np.hypot(*(b - a))) - 0.5  # A and B may already stand closer than that
    best = None
    for r in np.arange(args.side, 200, 5):
        for ang in np.radians(np.arange(0, 360, 10)):
            t = (a + b) / 2 + r * np.array([np.cos(ang), np.sin(ang)])
            if not (0 <= t[0] <= args.xmax and 0 <= t[1] <= args.ymax):
                continue
            if _dist_to_segment(t, b, a) < clear or \
                    min(_dist_to_segment(b, a, t), _dist_to_segment(a, t, b)) < near:
                continue
            cost = np.hypot(*(t - a)) + np.hypot(*(b - t))
            if best is None or cost < best[0]:
                best = (cost, t)
        if best is not None:
            return float(best[1][0]), float(best[1][1])
    raise RuntimeError(f"no free spot beside {a} and {b}")


def carry(b, frm, to, args):
    """Drag a checker. --wiggle: a small circle under it first, to catch one that is a bit
    off-centre. --overshoot: the checker lags behind the magnet, so stop a little past
    the target before letting go."""
    b.move(*frm)
    magnet_on(b)
    time.sleep(args.grab)
    if args.wiggle > 0:
        for ang in np.radians(np.arange(0, 360, 45)):
            b.move(frm[0] + args.wiggle * np.cos(ang), frm[1] + args.wiggle * np.sin(ang))
        b.move(*frm)
    d = np.subtract(to, frm)
    dist = float(np.hypot(*d))
    end = np.asarray(to, float) + (d / dist * args.overshoot if dist > 1 else 0)
    end = np.clip(end, 0, [args.xmax, args.ymax])
    b.move(float(end[0]), float(end[1]))
    b.magnet(False)
    time.sleep(0.2)


def swap(b, a, bb, args):
    t = free_spot(a, bb, args)
    print(f"  swap ({a[0]:.0f},{a[1]:.0f}) <-> ({bb[0]:.0f},{bb[1]:.0f}) via ({t[0]:.0f},{t[1]:.0f})")
    carry(b, a, t, args)
    carry(b, bb, a, args)
    carry(b, t, bb, args)


def main(args):
    c = calib.load()
    print(f"calib.json: error {c.rms_mm:.1f} mm RMS, worst {c.max_mm:.1f} mm")
    done = 0
    with Board(args.port) as b:
        b.home()
        if args.drag:
            b.drag_profile(*args.drag)
        while done < args.times:
            b.move(*args.park)
            time.sleep(0.3)
            img, _ = photo(b)
            found = detect(c, img, args.xmax, args.ymax)
            if args.auto and len(found) == 2:
                pair = [(f[2], f[3]) for f in found]
            else:
                pair = choose(c, img, found)
            if pair == "quit":
                break
            if pair == "again":
                continue
            swap(b, *pair, args)
            done += 1
        b.move(*args.park)
        b.magnet(False)
    cv2.destroyAllWindows()
    print(f"{done} swaps.")


if __name__ == "__main__":
    ui.keep_open()
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--times", type=int, default=1, help="how many swaps")
    ap.add_argument("--auto", action="store_true", help="no window when exactly 2 are found")
    ap.add_argument("--park", type=lambda t: tuple(map(float, t.split(","))), default=(415, 415),
                    help="where the magnet waits during the photo, x,y in mm")
    ap.add_argument("--side", type=float, default=40,
                    help="min mm from the middle of A-B to the waiting spot (grows if needed)")
    ap.add_argument("--grab", type=float, default=0.3, help="s to wait after the magnet turns on")
    ap.add_argument("--drag", type=lambda t: tuple(map(float, t.split(","))),
                    help="speed,accel while dragging, mm/s and mm/s^2 (firmware default 80,300)")
    ap.add_argument("--overshoot", type=float, default=3,
                    help="mm past the target before the magnet lets go (the checker lags)")
    ap.add_argument("--wiggle", type=float, default=0,
                    help="mm radius of a small circle under the checker after grabbing (0 = off)")
    ap.add_argument("--xmax", type=float, default=415)
    ap.add_argument("--ymax", type=float, default=420)
    main(ap.parse_args())
