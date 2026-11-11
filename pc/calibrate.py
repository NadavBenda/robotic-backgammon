"""The robot calibrates the camera by itself.

    py calibrate.py                  # measure a 6x6 grid and write calib.json
    py calibrate.py --n 5 --xmax 415 --ymax 420
    py calibrate.py test             # click a spot in the photo -> the magnet goes there

How it works: one checker rides on the magnet to each grid point. At every point the magnet
turns off, a photo is taken and the checker is found in it (dark ring, Hough circles).
A window shows the guess (green). Enter = accept, click = put the centre there yourself,
s = skip this point, q = stop. After the first two rows the guess uses the points so far, so most of
the time you only press Enter.
Photos are saved in captures/calib/.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import calib
from board import Board, BoardError

OUT = Path(__file__).parent / "captures" / "calib"
WIN = "calibrate"


def grid(n, xmax, ymax, margin):
    xs = np.linspace(margin, xmax - margin, n)
    ys = np.linspace(margin, ymax - margin, n)
    pts = []
    for j, y in enumerate(ys):  # serpentine: short moves
        row = xs if j % 2 == 0 else xs[::-1]
        pts += [(float(x), float(y)) for x in row]
    return pts


def photo(b):
    jpeg = b.capture()
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR), jpeg


def find_circles(img):
    g = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                         param1=100, param2=28, minRadius=6, maxRadius=70)
    return [] if c is None else [(float(x), float(y), float(r)) for x, y, r in c[0]]


def predict(px, mm, target, n):
    """Where the next point should be in the photo, from the points measured so far."""
    if len(px) < n + 2:  # need points from 2 rows, one row is a line
        return None
    H, _ = cv2.findHomography(np.float32(mm), np.float32(px))
    if H is None:
        return None
    p = H @ np.array([target[0], target[1], 1.0])
    return p[0] / p[2], p[1] / p[2]


def ask(img, guess, label):
    """Show the photo; return (u, v), 'skip' or 'quit'."""
    state = {"pt": guess}

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["pt"] = (float(x), float(y))

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse)
    while True:
        view = img.copy()
        if state["pt"]:
            u, v = map(int, state["pt"])
            cv2.circle(view, (u, v), 14, (0, 255, 0), 2)
            cv2.drawMarker(view, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 30, 1)
        cv2.putText(view, label + "   Enter=OK  click=fix  s=skip  q=stop", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(WIN, view)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 10) and state["pt"]:
            return state["pt"]
        if k == ord("s"):
            return "skip"
        if k == ord("q"):
            return "quit"


def magnet_on(b):
    while True:
        try:
            b.magnet(True)
            return
        except BoardError as e:
            if "MAG_HOT" not in str(e):
                raise
            print("   magnet hot, waiting 10 s...")
            time.sleep(10)


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    pts = grid(args.n, args.xmax, args.ymax, args.margin)
    px, mm = [], []
    with Board(args.port) as b:
        print("Homing...")
        b.home()
        b.move(*pts[0])
        input(f"Put ONE checker on the glass right above the magnet (at {pts[0]}), then Enter: ")
        for i, (x, y) in enumerate(pts):
            magnet_on(b)
            b.move(x, y)
            b.magnet(False)
            time.sleep(0.3)
            img, jpeg = photo(b)
            (OUT / f"p{i:02d}_{x:.0f}_{y:.0f}.jpg").write_bytes(jpeg)
            pred = predict(px, mm, (x, y), args.n)
            circles = find_circles(img)
            guess = None
            if circles:
                ref = pred or (img.shape[1] / 2, img.shape[0] / 2)
                c = min(circles, key=lambda c: np.hypot(c[0] - ref[0], c[1] - ref[1]))
                if pred is None or np.hypot(c[0] - pred[0], c[1] - pred[1]) < 60:
                    guess = (c[0], c[1])
            if guess is None and pred is not None:
                guess = pred
            if args.auto and pred is not None and guess != pred and \
                    np.hypot(guess[0] - pred[0], guess[1] - pred[1]) < 25:
                ans = guess
            else:
                ans = ask(img, guess, f"point {i + 1}/{len(pts)}  ({x:.0f}, {y:.0f}) mm")
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
    c = calib.fit(px, mm)
    c.save()
    print(f"\nSaved {calib.PATH.name}: {len(px)} points, error {c.rms_mm:.1f} mm RMS, worst {c.max_mm:.1f} mm")
    print("Goal: worst under 3 mm. Check it with:  py calibrate.py test")


def test(args):
    c = calib.load()
    print(f"calib.json: error {c.rms_mm:.1f} mm RMS, worst {c.max_mm:.1f} mm")
    with Board(args.port) as b:
        b.home()
        while True:
            img, _ = photo(b)
            ans = ask(img, None, "click a checker, Enter = go there, q = quit")
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
    ap.add_argument("--xmax", type=float, default=415)
    ap.add_argument("--ymax", type=float, default=420)
    ap.add_argument("--margin", type=float, default=15, help="mm from each edge")
    ap.add_argument("--auto", action="store_true", help="accept guesses that match the prediction")
    a = ap.parse_args()
    (test if a.mode == "test" else run)(a)
