"""Find the fastest drag that still holds the checker.

    py drag_test.py
    py drag_test.py --speeds 120,150,180,200 --accels 300,600,1000,1500

Click a checker, then click where to drag it (the longer the better, ~300 mm).
For every speed (120, 160, 200, 250) and acceleration (300, 800), low to high, the
magnet drags the checker once, A->B for the first profile, B->A for the next, and so
on, so it never has to be put back. After each drag the magnet steps --away mm
aside (like calibrate.py), takes a photo and you click where the checker really is
(no automatic detection; s = can't see it, then a photo from the corner). A drag
passes when it landed within --tol mm (30) of the target. Photos: captures/drag_test/. When an acceleration fails, higher
accelerations are skipped at that speed and at every faster one (at most 8 runs). If the checker is lost,
the window asks you to click where it is now.
Then click the next checker to test it too, or q to finish. At the end: a table per
checker, captures/drag_test.csv, and the profile that held for every checker tested,
left set on the board until the next reset.
"""
import sys  # noqa: E402
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent / "lib"))  # the internal code lives in lib/

import argparse
import csv
import time
from pathlib import Path

import numpy as np

import calib
import ui
from board import Board
from helpers import ask, photo, step_aside
from swap import carry

OUT = Path(__file__).parent / "captures"


def floats(t):
    return [float(x) for x in t.split(",")]


SHOTS = OUT / "drag_test"
DEFAULT_DRAG = (180, 800)  # same as board.ino, restored when nothing held


def where(b, c, args, target, label):
    """Photo with the magnet stepped aside; you click where the checker is.
    Returns its position in mm and the photo, or (None, photo) if you press s (not visible)."""
    SHOTS.mkdir(parents=True, exist_ok=True)
    b.move(*step_aside(*target, args))
    time.sleep(0.3)
    img, jpeg = photo(b)
    where.n = getattr(where, "n", 0) + 1
    (SHOTS / f"{where.n:03d}_to_{target[0]:.0f}_{target[1]:.0f}.jpg").write_bytes(jpeg)
    top, f = calib.view(img, c)   # shown corrected, seen from straight above
    ans = ask(top, None, label, (f.px_to_mm, f.mm_to_px),
              "click the checker, Enter=OK  s=can't see it  q=stop")
    if ans == "quit":
        return "quit", img
    if ans == "skip":
        return None, img
    return tuple(float(t) for t in f.px_to_mm(*ans)), img


def click_mm(c, img, label):
    top, f = calib.view(img, c)
    ans = ask(top, None, label, (f.px_to_mm, f.mm_to_px), "click it, Enter=OK  q=stop")
    if ans in ("quit", "skip"):
        return None
    return tuple(float(t) for t in f.px_to_mm(*ans))


def test_one(b, c, args, a_pt, b_pt):
    """Run the ladder for one checker. Returns (rows, best row or None)."""
    d = float(np.hypot(b_pt[0] - a_pt[0], b_pt[1] - a_pt[1]))
    print(f"\nA ({a_pt[0]:.0f},{a_pt[1]:.0f}) <-> B ({b_pt[0]:.0f},{b_pt[1]:.0f}), {d:.0f} mm")
    print(f"{'speed':>6} {'accel':>6}   leg   {'error':>6}  result")
    rows = []
    pos = a_pt  # where the checker is now (measured)
    target, leg = b_pt, "A->B"  # one drag per profile, direction alternates
    stop = False
    failed_at = None  # lowest accel that failed at a slower speed: no need to try it faster
    for v in args.speeds:
        for acc in args.accels:
            if failed_at is not None and acc >= failed_at:
                break
            b.drag_profile(v, acc)
            carry(b, pos, target, args)
            seen, img = where(b, c, args, target, f"{v:.0f} mm/s, {acc:.0f} mm/s2, {leg}: where is it now?")
            err = None
            if seen == "quit":
                stop = True
                break
            if seen is None:  # not visible from beside it: look from the corner
                b.move(*args.park)
                img, _ = photo(b)
                seen = click_mm(c, img, f"lost at {v:.0f} mm/s, {acc:.0f} mm/s2: where is it?")
                if seen is None:
                    stop = True
            else:
                err = float(np.hypot(seen[0] - target[0], seen[1] - target[1]))
            ok = err is not None and err <= args.tol
            print(f"{v:6.0f} {acc:6.0f}  {leg}  " + (f"{err:6.1f}" if err is not None else "  lost")
                  + f"  {'OK' if ok else 'FAIL'}")
            rows.append({"from": f"{a_pt[0]:.0f},{a_pt[1]:.0f}", "to": f"{b_pt[0]:.0f},{b_pt[1]:.0f}",
                         "speed": v, "accel": acc, "leg": leg, "err": err, "ok": ok})
            if stop:
                break
            pos = seen
            # next drag goes back the other way, from wherever it really is
            target, leg = (a_pt, "B->A") if target == b_pt else (b_pt, "A->B")
            if not ok:  # a higher accel at this speed, or at a faster one, will not hold
                failed_at = acc
                break
        if stop or failed_at == args.accels[0]:
            break
    good = [r for r in rows if r["ok"]]
    best = min(good, key=lambda r: move_time(d, r["speed"], r["accel"])) if good else None
    if best:
        print(f"Fastest that held here: {best['speed']:.0f} mm/s, {best['accel']:.0f} mm/s2 "
              f"({move_time(d, best['speed'], best['accel']):.2f} s for this drag)")
    else:
        print("Nothing held here.")
    return rows, best


def main(args):
    c = calib.load()
    rows, dists = [], []
    with Board(args.port) as b:
        b.home_if_needed()
        b.move(*args.park)
        n = 1
        while True:
            img, _ = photo(b)
            a_pt = click_mm(c, img, f"checker {n}: click the checker to drag (q = finish)")
            if a_pt is None:
                break
            b_pt = click_mm(c, img, f"checker {n}: click where to drag it (q = finish)")
            if b_pt is None:
                break
            rows += test_one(b, c, args, a_pt, b_pt)[0]
            dists.append(float(np.hypot(b_pt[0] - a_pt[0], b_pt[1] - a_pt[1])))
            n += 1
            b.move(*args.park)  # so the next photo shows the whole board
        # profiles that held for every checker tested (skipped = did not hold)
        okays = [{(r["speed"], r["accel"]) for r in rows if r["ok"] and r["from"] == f}
                 for f in dict.fromkeys(r["from"] for r in rows)]
        common = set.intersection(*okays) if okays else set()
        if common:
            v, a = min(common, key=lambda p: move_time(dists[0], *p))
            b.drag_profile(v, a)
            print(f"\nHeld for all {len(okays)} checkers, fastest: {v:.0f} mm/s, {a:.0f} mm/s2. "
                  f"Set on the board now (until the next reset); "
                  f"tell Claude to make it the default in board.ino.")
        elif okays:
            b.drag_profile(*DEFAULT_DRAG)
            print("\nNo profile held for every checker. Try lower accelerations.")

    if rows:
        OUT.mkdir(exist_ok=True)
        with open(OUT / "drag_test.csv", "w", newline="") as f:
            w = csv.DictWriter(f, ["from", "to", "speed", "accel", "leg", "err", "ok"])
            w.writeheader()
            w.writerows(rows)


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
    ap.add_argument("--speeds", type=floats, default=floats("120,160,200,250"),
                    help="mm/s, low to high (max 600)")
    ap.add_argument("--accels", type=floats, default=floats("300,800"),
                    help="mm/s^2, low to high (max 5000)")
    ap.add_argument("--tol", type=float, default=30, help="mm from the target that still counts as OK")
    ap.add_argument("--overshoot", type=float, default=5)
    ap.add_argument("--grab", type=float, default=0.3)
    ap.add_argument("--wiggle", type=float, default=0)
    ap.add_argument("--park", type=lambda t: tuple(map(float, t.split(","))), default=(205, 420),
                    help="where the magnet waits for the first photo and at the end")
    ap.add_argument("--away", type=float, default=80,
                    help="mm the magnet steps aside in each axis before a photo")
    ap.add_argument("--xmax", type=float, default=410)
    ap.add_argument("--ymax", type=float, default=420)
    main(ap.parse_args())
