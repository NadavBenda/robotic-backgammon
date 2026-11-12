"""The robot calibrates the camera by itself.

    py calibrate.py                  # measure a 6x6 grid and write calib.json
    py calibrate.py --n 5 --xmax 410 --ymax 420
    py calibrate.py test             # click a spot in the photo -> the magnet goes there

How it works: one checker rides on the magnet to each grid point. At every point the magnet
turns off, a photo is taken and the checker is found in it (dark ring, Hough circles).
A window shows the guess (green): the ellipse is where the round bottom of the checker
should appear at that spot (from calib.json if there is one, then from the points measured
so far). Click so the ellipse sits on the bottom of the checker. Enter = accept, s = skip
this point, q = stop.
Before each photo the magnet steps aside by --away mm in both axes (towards larger
coordinates, or smaller near the far edge) so the carriage and rail do not hide the checker.
After the first two rows the guess uses the points so far, so most of
the time you only press Enter.
Photos are saved in captures/calib/.
"""
import sys  # noqa: E402
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent / "lib"))  # the internal code lives in lib/

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import calib
import ui
from board import Board, BoardError
from helpers import ask, find_circles, magnet_on, photo, step_aside

OUT = Path(__file__).parent / "captures" / "calib"


def grid(n, xmax, ymax, margin):
    xs = np.linspace(margin, xmax - margin, n)
    ys = np.linspace(margin, ymax - margin, n)
    pts = []
    for j, y in enumerate(ys):  # serpentine: short moves
        row = xs if j % 2 == 0 else xs[::-1]
        pts += [(float(x), float(y)) for x in row]
    return pts






def predict(px, mm, target, n):
    """Where the next point should be in the photo, from the points measured so far."""
    if len(px) < n + 2:  # need points from 2 rows, one row is a line
        return None
    H, _ = cv2.findHomography(np.float32(mm), np.float32(px))
    if H is None:
        return None
    p = H @ np.array([target[0], target[1], 1.0])
    return p[0] / p[2], p[1] / p[2]


def live_maps(px, mm, n):
    """(to_mm, to_px) from the points measured so far (homography), or None."""
    if len(px) < n + 2:
        return None
    H, _ = cv2.findHomography(np.float32(mm), np.float32(px))
    if H is None:
        return None
    Hi = np.linalg.inv(H)
    return (lambda u, v: calib._homog(Hi, u, v)), (lambda x, y: calib._homog(H, x, y))


def saved_maps():
    """(to_mm, to_px) from an earlier calib.json, or None."""
    try:
        c = calib.load()
    except (OSError, ValueError, KeyError):
        return None
    return c








def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    pts = grid(args.n, args.xmax, args.ymax, args.margin)
    px, mm = [], []
    old = saved_maps()
    with Board(args.port) as b:
        b.home_if_needed()
        b.move(*pts[0])
        input(f"Put ONE checker on the glass right above the magnet (at {pts[0]}), then Enter: ")
        for i, (x, y) in enumerate(pts):
            magnet_on(b)
            b.move(x, y)
            b.magnet(False)
            b.move(*step_aside(x, y, args))  # so the carriage does not hide the checker
            time.sleep(0.3)
            img, jpeg = photo(b)
            b.move(x, y)  # back under the checker (magnet off), the next loop turns it on
            (OUT / f"p{i:02d}_{x:.0f}_{y:.0f}.jpg").write_bytes(jpeg)
            s = img.shape[1] / 1024
            if old is not None and tuple(old.size) != (img.shape[1], img.shape[0]):
                old = None  # made for another camera resolution: its ellipses would be wrong
            pred = predict(px, mm, (x, y), args.n)
            circles = find_circles(img)
            guess = None
            if circles:
                ref = pred or (img.shape[1] / 2, img.shape[0] / 2)
                c = min(circles, key=lambda c: np.hypot(c[0] - ref[0], c[1] - ref[1]))
                if pred is None or np.hypot(c[0] - pred[0], c[1] - pred[1]) < 60 * s:
                    guess = (c[0], c[1])
            if guess is None and pred is not None:
                guess = pred
            if args.auto and pred is not None and guess != pred and \
                    np.hypot(guess[0] - pred[0], guess[1] - pred[1]) < 25 * s:
                ans = guess
            else:
                ans = ask(img, guess, f"point {i + 1}/{len(pts)}  ({x:.0f}, {y:.0f}) mm",
                          live_maps(px, mm, args.n) or (old and (old.px_to_mm, old.mm_to_px)))
            if ans == "quit":
                break
            if ans == "skip":
                continue
            px.append(ans)
            mm.append((x, y))
            print(f"  {i + 1:2d}/{len(pts)}  ({x:5.1f},{y:5.1f}) mm  ->  ({ans[0]:6.1f},{ans[1]:6.1f}) px")
        b.magnet(False)
    cv2.destroyAllWindows()

    if len(px) < 12:
        sys.exit(f"Only {len(px)} points, need at least 12 for the fit.")
    c = calib.fit(px, mm, size=(img.shape[1], img.shape[0]))
    c.save()
    print(f"\nSaved {calib.PATH.name}: {len(px)} points, error {c.rms_mm:.1f} mm RMS, worst {c.max_mm:.1f} mm")
    print("Goal: worst under 3 mm. Check it with:  py calibrate.py test")


def test(args):
    c = calib.load()
    print(f"calib.json: error {c.rms_mm:.1f} mm RMS, worst {c.max_mm:.1f} mm")
    with Board(args.port) as b:
        b.home_if_needed()
        while True:
            b.move(*args.park)
            img, _ = photo(b)
            c.check(img)
            ans = ask(img, None, "test: click a checker", (c.px_to_mm, c.mm_to_px),
                      "Enter=go there  q=quit")
            if ans in ("quit", "skip"):
                break
            x, y = (float(v) for v in c.px_to_mm(*ans))
            print(f"  ({ans[0]:.0f},{ans[1]:.0f}) px -> G {x:.1f} {y:.1f}")
            try:
                b.move(x, y)
            except BoardError as e:
                print("  ", e)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="run", choices=["run", "test"])
    ap.add_argument("--port")
    ap.add_argument("--n", type=int, default=6, help="grid size n x n")
    ap.add_argument("--xmax", type=float, default=410)
    ap.add_argument("--ymax", type=float, default=420)
    ap.add_argument("--margin", type=float, default=15, help="mm from each edge")
    ap.add_argument("--away", type=float, default=80,
                    help="mm the magnet steps aside in each axis before a photo")
    ap.add_argument("--park", type=lambda t: tuple(map(float, t.split(","))), default=(410, 420),
                    help="test mode: where the magnet waits during the photo, x,y in mm")
    ap.add_argument("--auto", action="store_true", help="accept guesses that match the prediction")
    ui.keep_open()
    a = ap.parse_args()
    (test if a.mode == "test" else run)(a)
