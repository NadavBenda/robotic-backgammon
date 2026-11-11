"""Find the fastest drag that still holds the checker.

    py drag_test.py
    py drag_test.py --speeds 120,150,180,200 --accels 300,600,1000,1500

Click a checker, then click where to drag it (the longer the better, ~200 mm).
For every acceleration, from low to high, and every speed, from low to high, the magnet
drags the checker there and back. After each leg a photo measures where the checker
really landed. A leg passes when it landed within --tol mm of the target. When a speed
fails, the faster speeds at that acceleration are skipped. If the checker is lost,
the window asks you to click where it is now.
At the end: a table, captures/drag_test.csv, and the fastest profile that passed,
which is also left set on the board (until the next reset). Put it in board.ino
(dragMaxSteps / dragAccelSteps) or use it with  py swap.py --drag v,a
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

import calib
import ui
from board import Board
from calibrate import ask, photo
from swap import carry, detect

OUT = Path(__file__).parent / "captures"


def floats(t):
    return [float(x) for x in t.split(",")]


def where(b, c, args, near):
    """Photo from --park; the checker closest to `near` in mm, or None if none within 40 mm."""
    b.move(*args.park)
    time.sleep(0.3)
    img, _ = photo(b)
    found = detect(c, img, args.xmax, args.ymax)
    if found:
        f = min(found, key=lambda f: np.hypot(f[2] - near[0], f[3] - near[1]))
        if np.hypot(f[2] - near[0], f[3] - near[1]) < 40:
            return (f[2], f[3]), img
    return None, img


def click_mm(c, img, label):
    ans = ask(img, None, label, (c.px_to_mm, c.mm_to_px), "click it, Enter=OK  q=stop")
    if ans in ("quit", "skip"):
        return None
    return tuple(float(t) for t in c.px_to_mm(*ans))


def main(args):
    c = calib.load()
    rows, best = [], None
    with Board(args.port) as b:
        b.home()
        b.move(*args.park)
        img, _ = photo(b)
        a_pt = click_mm(c, img, "1/2: the checker to drag")
        if a_pt is None:
            return
        b_pt = click_mm(c, img, "2/2: where to drag it")
        if b_pt is None:
            return
        print(f"A ({a_pt[0]:.0f},{a_pt[1]:.0f}) <-> B ({b_pt[0]:.0f},{b_pt[1]:.0f}), "
              f"{np.hypot(b_pt[0] - a_pt[0], b_pt[1] - a_pt[1]):.0f} mm")
        print(f"{'accel':>6} {'speed':>6} {'A->B err':>9} {'B->A err':>9}  result")
        pos = a_pt  # where the checker is now (measured)
        stop = False
        for acc in args.accels:
            for v in args.speeds:
                b.drag_profile(v, acc)
                errs = []
                for target in (b_pt, a_pt):
                    carry(b, pos, target, args)
                    seen, img = where(b, c, args, target)
                    if seen is None:
                        errs.append(None)
                        seen = click_mm(c, img, f"lost at {v:.0f} mm/s, {acc:.0f} mm/s2: where is it?")
                        if seen is None:
                            stop = True
                        pos = seen
                        break
                    errs.append(float(np.hypot(seen[0] - target[0], seen[1] - target[1])))
                    pos = seen
                ok = len(errs) == 2 and all(e is not None and e <= args.tol for e in errs)
                cells = [f"{e:9.1f}" if e is not None else "     lost" for e in errs]
                cells += ["        -"] * (2 - len(cells))
                print(f"{acc:6.0f} {v:6.0f} {cells[0]} {cells[1]}  {'OK' if ok else 'FAIL'}")
                rows.append({"accel": acc, "speed": v, "err_ab": errs[0] if errs else None,
                             "err_ba": errs[1] if len(errs) > 1 else None, "ok": ok})
                if stop or not ok:
                    break  # faster speeds at this accel will not do better
            if stop:
                break
        good = [r for r in rows if r["ok"]]
        best = None
        if good:  # "fastest" = shortest move time over the test distance
            d = np.hypot(b_pt[0] - a_pt[0], b_pt[1] - a_pt[1])
            best = min(good, key=lambda r: move_time(d, r["speed"], r["accel"]))
            b.drag_profile(best["speed"], best["accel"])
        b.move(*args.park)

    OUT.mkdir(exist_ok=True)
    with open(OUT / "drag_test.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["accel", "speed", "err_ab", "err_ba", "ok"])
        w.writeheader()
        w.writerows(rows)
    if best:
        t = move_time(np.hypot(b_pt[0] - a_pt[0], b_pt[1] - a_pt[1]), best["speed"], best["accel"])
        print(f"\nFastest that held: {best['speed']:.0f} mm/s, {best['accel']:.0f} mm/s2 "
              f"({t:.2f} s for this drag). Set on the board now; for good put it in board.ino "
              f"or run  py swap.py --drag {best['speed']:.0f},{best['accel']:.0f}")
    else:
        print("\nNothing passed. Try lower --accels or a larger --tol.")


def move_time(d, v, a, v0=20.0):
    """Seconds for a trapezoid move of d mm (start speed v0 like the firmware)."""
    ramp = (v * v - v0 * v0) / (2 * a)  # mm to reach v
    if 2 * ramp >= d:  # never reaches v: triangle
        vp = np.sqrt(v0 * v0 + a * d)
        return 2 * (vp - v0) / a
    return 2 * (v - v0) / a + (d - 2 * ramp) / v


if __name__ == "__main__":
    ui.keep_open()
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--speeds", type=floats, default=floats("100,120,140,160,180,200"),
                    help="mm/s, low to high (max 200)")
    ap.add_argument("--accels", type=floats, default=floats("300,500,800,1200,2000"),
                    help="mm/s^2, low to high (max 2000)")
    ap.add_argument("--tol", type=float, default=4, help="mm from the target that still counts as OK")
    ap.add_argument("--overshoot", type=float, default=3)
    ap.add_argument("--grab", type=float, default=0.3)
    ap.add_argument("--wiggle", type=float, default=0)
    ap.add_argument("--park", type=lambda t: tuple(map(float, t.split(","))), default=(415, 415))
    ap.add_argument("--xmax", type=float, default=415)
    ap.add_argument("--ymax", type=float, default=420)
    main(ap.parse_args())
